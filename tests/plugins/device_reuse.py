# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reuse one opened device across the tests of a pytest session.

Without this, the function-scoped ``device`` fixture opens and closes the device around every
parametrized case: ~0.1 s per reopen plus the close, and on a whole golden or unit-test file
that is most of the session (measured: 205 opens = 20 s of a 36 s run).

How: ``ttnn.CreateDevice`` returns the device already open when it is called again with the same
arguments and the same TT_METAL_*/TTNN_* environment (the runtime reads those at open), and
``ttnn.close_device`` on that device only synchronizes it. Fixtures and tests are unchanged.

A device is handed out again only if it is in the state a fresh open would give:
  - the previous test passed (or was skipped). After any failed or xfailed test the device is
    closed and opened again: a hang, a dispatch timeout or a crash mid-op can leave a device
    whose synchronize still succeeds;
  - it is still initialized (nothing closed it another way, e.g. ttnn.CloseDevice or
    device.close());
  - after the reset below, no buffers are allocated on it in DRAM, L1, L1_SMALL or TRACE.
Reset before reuse: program cache enabled and emptied, cache misses allowed, no sub-device
manager loaded. Kernel binaries stay in the in-process JIT build cache, as they did across tests
of one process already.

Every other way to open a device (ttnn.open_device, open_mesh_device, CreateDevices) first
closes the kept device unless a test still holds it, so it never runs next to a device that would
have been closed already. A close through ttnn.CloseDevice / CloseDevices / close_mesh_device
drops the kept device.

Known difference: a test that calls ttnn.close_device and then keeps using the device works
here (the device stays open), where it would fail on a fresh device.

Loaded by scripts/run_safe_pytest.sh (``-p tests.plugins.device_reuse``) on hardware, unless
SAFE_PYTEST_REUSE_DEVICE=0. With SAFE_PYTEST_KEEP_DEVICE_OPEN=1 (set by
scripts/safe_pytest_warm.py in its long-lived process) the device stays open at the end of the
session for the next one; ``release()`` closes it.
"""

from __future__ import annotations

import gc
import inspect
import os
import sys

import pytest

# Modules whose attributes the fixtures and helpers call (ttnn.manage_device uses ttnn.device's).
_MODULES = ("ttnn", "ttnn.device", "ttnn.distributed.distributed")
_CACHED_OPEN = "CreateDevice"
_OTHER_OPENS = ("open_device", "open_mesh_device", "CreateDevices")
_RELEASE = "close_device"  # keeps the device
_REAL_CLOSES = ("CloseDevice", "CloseDevices", "close_mesh_device")  # drop it
_BUFFER_TYPES = ("DRAM", "L1", "L1_SMALL", "TRACE")
# Read on every use, not at device open; the precompile plugin toggles them mid-session.
_LAZY_ENV = ("TT_METAL_JIT_SERVER_", "TT_METAL_JIT_PREPROCESS")

_originals: dict[tuple[str, str], object] = {}
_cached = {"key": None, "device": None, "baseline": None, "in_use": False, "poisoned": False}
_stats = {"opens": 0, "reuses": 0, "reopens_dirty": 0}


def _ttnn():
    return sys.modules["ttnn"]


def _original(name: str):
    for module_name in _MODULES:
        fn = _originals.get((module_name, name))
        if fn is not None:
            return fn
    return getattr(_ttnn(), name)


def _value(name: str, value):
    """Comparable form of one argument, or None if it cannot be compared by value."""
    if name == "dispatch_core_config":
        if value is None:
            # CreateDevice(dispatch_core_config=None) opens with the arch default, the same
            # config ttnn.DispatchCoreConfig() resolves to (the C++ config is type + axis).
            value = _ttnn().DispatchCoreConfig()
        return f"DispatchCoreConfig({value.type!r}, {value.axis!r})"
    text = repr(value)
    return None if " at 0x" in text else text


def _key(fn, args, kwargs):
    """Hashable identity of the requested device: the call bound to fn's signature with its
    defaults filled in, plus the TT_METAL_*/TTNN_* environment. None if an argument cannot be
    compared by value (then the call is passed through and nothing is kept)."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    bound.apply_defaults()
    parts = []
    for name, value in bound.arguments.items():
        text = _value(name, value)
        if text is None:
            return None
        parts.append(f"{name}={text}")
    env = sorted(
        (k, v)
        for k, v in os.environ.items()
        if k.startswith(("TT_METAL_", "TTNN_")) and not k.startswith(_LAZY_ENV)
    )
    return (tuple(parts), tuple(env))


def _allocated(device):
    view = _ttnn()._ttnn.device.GetMemoryView
    out = []
    for name in _BUFFER_TYPES:
        buffer_type = getattr(_ttnn().BufferType, name, None)
        if buffer_type is not None:
            out.append(view(device, buffer_type).total_bytes_allocated_per_bank)
    return tuple(out)


def _reset(device) -> None:
    device.enable_program_cache()
    device.clear_program_cache()
    if hasattr(device, "set_program_cache_misses_allowed"):
        device.set_program_cache_misses_allowed(True)
    device.clear_loaded_sub_device_manager()


def _forget() -> None:
    _cached.update(key=None, device=None, baseline=None, in_use=False, poisoned=False)


def _close_cached() -> None:
    device = _cached["device"]
    _forget()
    if device is not None:
        try:
            initialized = device.is_initialized()
        except Exception:  # noqa: BLE001 - a broken handle: nothing left to close
            initialized = False
        if initialized:
            _original(_RELEASE)(device)


def release() -> None:
    """Close the kept device for real (end of session, or the warm process shutting down)."""
    try:
        _close_cached()
    except Exception:  # noqa: BLE001 - a broken device must not mask the session result
        pass


def _reusable(device) -> bool:
    if _cached["poisoned"]:
        return False
    try:
        if not device.is_initialized():
            _forget()  # closed another way; nothing to close
            return False
        _reset(device)
        # A full collection only when something is still allocated: tensors in reference cycles
        # are freed by it, and it is too slow to run before every test.
        if _allocated(device) == _cached["baseline"]:
            return True
        gc.collect()
        return _allocated(device) == _cached["baseline"]
    except Exception:  # noqa: BLE001 - any doubt about the device means open a fresh one
        return False


def _wrap_cached_open(fn):
    def create_reused(*args, **kwargs):
        key = _key(fn, args, kwargs)
        if key is not None and key == _cached["key"] and not _cached["in_use"]:
            if _reusable(_cached["device"]):
                _cached["in_use"] = True
                _stats["reuses"] += 1
                return _cached["device"]
            _stats["reopens_dirty"] += 1
        if not _cached["in_use"]:
            _close_cached()
        device = fn(*args, **kwargs)
        _stats["opens"] += 1
        if key is not None and _cached["device"] is None:
            _cached.update(key=key, device=device, baseline=_allocated(device), in_use=True)
        return device

    return create_reused


def _wrap_other_open(fn):
    def open_other(*args, **kwargs):
        # A fresh-device session would have closed the kept device by now, unless a test holds it.
        if not _cached["in_use"]:
            _close_cached()
        return fn(*args, **kwargs)

    return open_other


def _wrap_release(fn):
    def close_reused(device, *args, **kwargs):
        if device is not None and device is _cached["device"]:
            try:
                _ttnn().synchronize_device(device)
                _cached["in_use"] = False
            except Exception:  # noqa: BLE001 - same as close_device: still close, then report
                _forget()
                fn(device, *args, **kwargs)
            return None
        return fn(device, *args, **kwargs)

    return close_reused


def _wrap_real_close(fn):
    def close_real(devices, *args, **kwargs):
        kept = _cached["device"]
        if kept is not None:
            values = devices.values() if isinstance(devices, dict) else [devices]
            if any(d is kept for d in values):
                _forget()
        return fn(devices, *args, **kwargs)

    return close_real


def install() -> None:
    """Wrap the device entry points (idempotent)."""
    for module_name in _MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name, wrap in (
            [(_CACHED_OPEN, _wrap_cached_open), (_RELEASE, _wrap_release)]
            + [(n, _wrap_other_open) for n in _OTHER_OPENS]
            + [(n, _wrap_real_close) for n in _REAL_CLOSES]
        ):
            fn = getattr(module, name, None)
            if fn is None or hasattr(fn, "_device_reuse_original"):
                continue
            _originals[(module_name, name)] = fn
            wrapped = wrap(fn)
            wrapped._device_reuse_original = fn
            setattr(module, name, wrapped)


def _enabled(config) -> bool:
    # xdist workers run their own sessions; only reuse within a plain (non-distributed) session.
    return os.environ.get("SAFE_PYTEST_REUSE_DEVICE", "1") != "0" and not getattr(
        config.option, "numprocesses", None
    )


# tryfirst: wrap before tests.plugins.phase_timing does, so its timer sits outside and sees every
# open call (reused or not); the split comes from _stats.
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not _enabled(config):
        return
    import ttnn  # noqa: F401 - the entry points live there

    _stats.update(opens=0, reuses=0, reopens_dirty=0)
    install()


def pytest_runtest_logreport(report):
    # A failed or xfailed test may have left the device hung or half-way through an op.
    if report.failed or getattr(report, "wasxfail", None) is not None:
        _cached["poisoned"] = True


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if not _enabled(config):
        return
    if os.environ.get("SAFE_PYTEST_KEEP_DEVICE_OPEN") != "1":
        release()
