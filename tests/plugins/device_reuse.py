# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reuse one opened device across the tests of a pytest session.

Without this, the function-scoped ``device`` fixture opens and closes the device around every
parametrized case: ~0.1 s per reopen plus the close, and on a whole golden or unit-test file
that is most of the session (measured: 205 opens = 20 s of a 36 s run).

How: ``ttnn.CreateDevice`` / ``ttnn.open_device`` return the device already open when they are
called again with the same arguments, and ``ttnn.close_device`` on that device only
synchronizes it. Fixtures and tests are unchanged. The device is really closed when a call asks
for different arguments (another l1_small_size, another device id), and at the end of the
session.

Each handed-out device starts like a freshly opened one, as far as tests can see:
  - program cache enabled, emptied, cache misses allowed (the defaults of a new device);
  - no sub-device manager loaded;
  - no buffers allocated. If buffers are still allocated after the previous test released its
    device (something kept a tensor alive), the device is closed and opened again instead.
Kernel binaries stay in the in-process JIT build cache, as they did across tests of one
process already.

A synchronize that fails on release (hang, dispatch timeout) closes the device for real and
drops it, exactly as the plain close_device would have.

Loaded by scripts/run_safe_pytest.sh (``-p tests.plugins.device_reuse``) on hardware, unless
SAFE_PYTEST_REUSE_DEVICE=0. Mesh devices (``open_mesh_device``) are not touched.

With SAFE_PYTEST_KEEP_DEVICE_OPEN=1 (set by scripts/safe_pytest_warm.py in its long-lived
process), the device stays open at the end of the session for the next one; ``release()`` closes it.
"""

from __future__ import annotations

import gc
import inspect
import os
import sys

import pytest

_OPEN_FNS = ("CreateDevice", "open_device")
_CLOSE_FNS = ("close_device",)
# Python-level modules that expose the entry points (ttnn.manage_device calls the ttnn.device ones).
_MODULES = ("ttnn", "ttnn.device")
_BUFFER_TYPES = ("DRAM", "L1", "L1_SMALL", "TRACE")

_originals: dict[tuple[str, str], object] = {}
_cached = {"key": None, "device": None, "baseline": None}
_stats = {"opens": 0, "reuses": 0, "reopens_dirty": 0}


def _ttnn():
    return sys.modules["ttnn"]


def _real(name: str):
    return _originals.get(("ttnn.device", name)) or _originals.get(("ttnn", name))


def _value(name: str, value):
    """Comparable form of one argument, or None if it cannot be compared by value."""
    if name == "dispatch_core_config":
        if value is None:
            # CreateDevice(dispatch_core_config=None) opens with the arch default, the same
            # config ttnn.DispatchCoreConfig() resolves to.
            value = _ttnn().DispatchCoreConfig()
        return f"DispatchCoreConfig({value.type!r}, {value.axis!r})"
    text = repr(value)
    return None if " at 0x" in text else text


def _key(fn_name: str, fn, args, kwargs):
    """Hashable identity of the requested device: the call bound to fn's signature with its
    defaults filled in, so CreateDevice(0) and CreateDevice(device_id=0, l1_small_size=<default>)
    match. None if an argument cannot be compared by value (then the call is passed through and
    nothing is cached)."""
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
    # open_device and CreateDevice open the same kind of device; key on the arguments only.
    return tuple(parts)


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


def _close_cached() -> None:
    device = _cached["device"]
    _cached.update(key=None, device=None, baseline=None)
    if device is not None:
        _real("close_device")(device)


def release() -> None:
    """Close the kept device for real (end of session, or the warm process shutting down)."""
    try:
        _close_cached()
    except Exception:  # noqa: BLE001 - a broken device must not mask the session result
        pass


def _make_open(fn_name: str, fn):
    def open_reused(*args, **kwargs):
        key = _key(fn_name, fn, args, kwargs)
        if key is not None and key == _cached["key"]:
            device = _cached["device"]
            # Reset first: programs held by the program cache can own device buffers.
            _reset(device)
            # A full collection only when something is still allocated: tensors in reference
            # cycles are freed by it, and it is too slow to run before every test.
            clean = _allocated(device) == _cached["baseline"]
            if not clean:
                gc.collect()
                clean = _allocated(device) == _cached["baseline"]
            if clean:
                _stats["reuses"] += 1
                return device
            _stats["reopens_dirty"] += 1
        _close_cached()
        device = fn(*args, **kwargs)
        _stats["opens"] += 1
        if key is not None:
            _cached.update(key=key, device=device, baseline=_allocated(device))
        return device

    open_reused._device_reuse_original = fn
    return open_reused


def _make_close(fn):
    def close_reused(device, *args, **kwargs):
        if device is not None and device is _cached["device"]:
            try:
                _ttnn().synchronize_device(device)
            except Exception:  # noqa: BLE001 - same as close_device: still close, then report
                _cached.update(key=None, device=None, baseline=None)
                fn(device, *args, **kwargs)
            return None
        return fn(device, *args, **kwargs)

    close_reused._device_reuse_original = fn
    return close_reused


def install() -> None:
    """Wrap the open/close entry points (idempotent)."""
    for module_name in _MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name in _OPEN_FNS + _CLOSE_FNS:
            fn = getattr(module, name, None)
            if fn is None or hasattr(fn, "_device_reuse_original"):
                continue
            _originals[(module_name, name)] = fn
            wrapped = _make_open(name, fn) if name in _OPEN_FNS else _make_close(fn)
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


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if not _enabled(config):
        return
    if os.environ.get("SAFE_PYTEST_KEEP_DEVICE_OPEN") != "1":
        release()
