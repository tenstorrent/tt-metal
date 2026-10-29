# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-phase wall-clock timings of one pytest process, for scripts/run_safe_pytest.sh.

Splits a run into interpreter + pytest startup, initial conftest load (where torch and ttnn
are usually imported), collection, device open and close, and the test loop, so the fixed
cost every invocation pays can be told apart from the tests themselves.

Writes nothing unless SAFE_PYTEST_PHASES_FILE is set. The file gets one JSON fragment of
integer milliseconds (no braces), which run_safe_pytest.sh splices into its device-timing
record:

    "phase_startup_ms":812,"phase_conftest_ms":2410,...

SAFE_PYTEST_LAUNCH_MS (epoch ms, set by the wrapper just before it launches pytest) anchors
phase_startup_ms and phase_unconfigure_at_ms; without it both are -1. Fields the controller
cannot see under pytest-xdist (collection and device time happen in the workers) are -1 too.
A process that dies before pytest_unconfigure (hang, kill) writes no fragment at all.

Device time counts calls made through ttnn.<fn>, ttnn.device.<fn> and
ttnn.distributed.distributed.<fn>, including names a test module imported from those modules
after initial conftests loaded. A name bound before that (or reached some other way) runs
unwrapped, and its time stays in phase_testloop_ms.

With precompile on (tests.plugins.up_front_collect), phase_testloop_ms includes the collect
pass and compile step that run_safe_pytest.sh reports as precompile_s, and the compile step's
own device open is counted in phase_device_opens; do not add precompile_s to it.

    SAFE_PYTEST_PHASES_FILE=/tmp/p.json PYTHONPATH="$PWD" pytest -p tests.plugins.phase_timing <tests>
"""

from __future__ import annotations

import functools
import os
import sys
import time

import pytest

_LOADED_S = time.time()
_OUT = os.environ.get("SAFE_PYTEST_PHASES_FILE", "")
_LAUNCH_MS = int(os.environ.get("SAFE_PYTEST_LAUNCH_MS", "-1") or -1)

# Device open/close entry points, and the modules that call them by their module-local names
# (ttnn.manage_device -> ttnn.device.open_device, create_mesh_device -> open_mesh_device).
_OPEN_FNS = ("CreateDevice", "open_device", "open_mesh_device")
_CLOSE_FNS = ("close_device", "close_mesh_device")
_DEVICE_MODULES = ("ttnn", "ttnn.device", "ttnn.distributed.distributed")

_ms = {
    "conftest": 0,
    "collect": 0,
    "ttnn_import": 0,
    "device_open": 0,
    "device_close": 0,
    "testloop": 0,
}
_device_opens = 0
# Outermost-call guard: one wrapped entry point may call another (open_device -> CreateDevice).
_depth = 0


def _elapsed_ms(start_s: float) -> int:
    return int(round((time.time() - start_s) * 1000))


def _timed(fn, key: str):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _depth, _device_opens
        if _depth:
            return fn(*args, **kwargs)
        _depth += 1
        start = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            _depth -= 1
            _ms[key] += _elapsed_ms(start)
            if key == "device_open":
                _device_opens += 1

    wrapper._phase_timing_original = fn
    return wrapper


def _patch_ttnn(import_if_missing: bool):
    """Wrap ttnn's device open/close entry points. With import_if_missing, imports ttnn if
    nothing has yet and books that import separately so it does not land in the test loop."""
    if "ttnn" not in sys.modules:
        if not import_if_missing:
            return
        start = time.time()
        try:
            import ttnn  # noqa: F401
        except ImportError:
            return
        _ms["ttnn_import"] = _elapsed_ms(start)
    for module_name in _DEVICE_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name, key in [(n, "device_open") for n in _OPEN_FNS] + [(n, "device_close") for n in _CLOSE_FNS]:
            fn = getattr(module, name, None)
            if callable(fn) and not hasattr(fn, "_phase_timing_original"):
                setattr(module, name, _timed(fn, key))


def _active(config) -> bool:
    # Under xdist only the controller reports; workers would overwrite each other.
    return bool(_OUT) and not hasattr(config, "workerinput")


def _distributed(config) -> bool:
    # Controller of an xdist run: collection and device work happen in the workers.
    return bool(getattr(config.option, "numprocesses", None))


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # Before collection, so `from ttnn import open_device` in a test module binds the wrapper.
    if _active(config):
        _patch_ttnn(import_if_missing=False)


@pytest.hookimpl(wrapper=True)
def pytest_load_initial_conftests(early_config, parser, args):
    start = time.time()
    try:
        return (yield)
    finally:
        _ms["conftest"] = _elapsed_ms(start)


@pytest.hookimpl(wrapper=True)
def pytest_collection(session):
    start = time.time()
    try:
        return (yield)
    finally:
        _ms["collect"] = _elapsed_ms(start)


@pytest.hookimpl(wrapper=True)
def pytest_runtestloop(session):
    if _active(session.config):
        _patch_ttnn(import_if_missing=True)
    start = time.time()
    try:
        return (yield)
    finally:
        _ms["testloop"] = _elapsed_ms(start)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if not _active(config):
        return
    now_ms = int(time.time() * 1000)
    anchored = _LAUNCH_MS >= 0
    in_workers = _distributed(config)
    fields = {
        "phase_startup_ms": int(_LOADED_S * 1000) - _LAUNCH_MS if anchored else -1,
        "phase_conftest_ms": _ms["conftest"],
        "phase_collect_ms": -1 if in_workers else _ms["collect"],
        "phase_ttnn_import_ms": _ms["ttnn_import"],
        "phase_device_open_ms": -1 if in_workers else _ms["device_open"],
        "phase_device_close_ms": -1 if in_workers else _ms["device_close"],
        "phase_device_opens": -1 if in_workers else _device_opens,
        "phase_testloop_ms": _ms["testloop"],
        "phase_unconfigure_at_ms": now_ms - _LAUNCH_MS if anchored else -1,
    }
    try:
        with open(_OUT, "w") as f:
            f.write(",".join(f'"{k}":{int(v)}' for k, v in fields.items()))
    except OSError:
        pass
