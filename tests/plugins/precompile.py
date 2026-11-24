# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Up-front precompile for a ttnn pytest run.

Before the tests run for real, every test body runs once in a dry run (ttnn.device.BeginDryRun): nothing is sent to
the device, and each program the body would have launched has its kernel builds queued in the background. When the
real pass starts, the kernels are in Metal's kernel cache and the tests do not wait for them.

    pytest -p tests.plugins.precompile <tests>          PRECOMPILE=on|off|auto (default auto)

The dry pass uses each test's own fixtures, so devices open and close as the tests ask. Its test results are never
reported. Whatever a body does differently without real device data only changes which kernels are built ahead: a
kernel that was not found is built by the test that needs it, as without this plugin.
"""

import gc
import os
import random
import signal
import time

import pytest
from _pytest.runner import runtestprotocol

_MODE = os.environ.get("PRECOMPILE", "auto")
# A dry body that runs longer than this is ended (a loop driven by device data that was never read, for example).
_BODY_SECONDS = float(os.environ.get("PRECOMPILE_BODY_SECONDS", "60"))
_STATE = {"dry": False, "bodies": 0, "ended_early": 0, "timed_out": 0}


# Calls that reach the device outside the command queue, or wait for an answer no blocked op will send: a dry body ends
# where it reaches one. The kernels of what came before are already queued.
_STOP_AT = {
    "ttnn": (
        "light_metal_begin_capture LightMetalReplay MeshSocket create_socket_pair H2DSocket D2HSocket H2DStreamService "
        "D2HStreamService D2DStreamService D2DStreamServiceSender D2DStreamServiceReceiver ReadDeviceProfiler"
    ),
    "ttnn.experimental": (
        "start_tensor_prefetcher queue_tensor_prefetcher_request queue_tensor_prefetcher_wait_for_signal "
        "wait_for_cq_on_tensor_prefetcher signal_tensor_prefetcher stop_tensor_prefetcher CommandListBuilder CommandList"
    ),
    "ttnn._experimental.reload": "read_core_l1 write_core_l1",
    "tt_py_test_utils_common.realtime_profiler_utils": "profile_realtime_program profile_realtime_program_merged",
}


class _BodyTimeout(BaseException):
    pass


class _BodyStop(BaseException):
    pass


def _stop(*args, **kwargs):
    raise _BodyStop()


def _swap_stops(saved):
    """With saved=None: replace every listed call by one that ends the body, and return what was replaced. With that
    list: put the real ones back."""
    import importlib

    if saved is not None:
        for module, name, real in saved:
            setattr(module, name, real)
        return None
    saved = []
    for module_name, names in _STOP_AT.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in names.split():
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, _stop)
    return saved


def _decide(session):
    """Whether this run gets a dry pass. Decided before it starts, from this run alone."""
    if _MODE in ("on", "off"):
        return _MODE == "on", f"PRECOMPILE={_MODE}"
    return False, "auto: no rule yet"


def _kernels_in_cache():
    """Number of kernels (kernels/<name>/<hash>) in Metal's kernel cache."""
    root = os.environ.get("TT_METAL_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "tt-metal-cache")
    count = 0
    try:
        for build_key in os.scandir(root):
            kernels = os.path.join(build_key.path, "kernels")
            if os.path.isdir(kernels):
                for name in os.scandir(kernels):
                    if name.is_dir():
                        count += sum(1 for _ in os.scandir(name.path))
    except OSError:
        pass
    return count


def _open_fabric_without_routers(ttnn):
    """Replace ttnn.set_fabric_config for the dry pass: a mesh opens without starting the fabric routers, so the open
    does not wait for them to sync. Nothing is dispatched in a dry run, so they would have no work. Returns the real
    function. A call that names a manager mode other than DEFAULT is passed through unchanged."""
    real = ttnn.set_fabric_config
    default, enabled = ttnn.FabricManagerMode.DEFAULT, ttnn.FabricManagerMode.ENABLED
    position = 5  # set_fabric_config(config, reliability_mode, num_planes, tensix, udm, manager, router)

    def set_fabric_config(*args, **kwargs):
        if getattr(args[0] if args else kwargs.get("config"), "name", None) != "DISABLED":
            if len(args) > position:
                if args[position] == default:
                    args = args[:position] + (enabled,) + args[position + 1 :]
            elif kwargs.get("fabric_manager_mode", default) == default:
                kwargs = {**kwargs, "fabric_manager_mode": enabled}
        return real(*args, **kwargs)

    ttnn.set_fabric_config = set_fabric_config
    return real


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    if not _STATE["dry"]:
        return (yield)
    import ttnn

    def on_alarm(signum, frame):
        raise _BodyTimeout()

    _STATE["bodies"] += 1
    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, _BODY_SECONDS)
    ttnn._ttnn.device.BeginDryRun()
    try:
        yield
    except KeyboardInterrupt:
        raise
    except _BodyTimeout:
        _STATE["timed_out"] += 1
    except BaseException:
        # A failed check, a skip, or an error from reading data the device never produced: the body ends here.
        _STATE["ended_early"] += 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        ttnn._ttnn.device.EndDryRun()
    # What the body left behind must be gone before its fixtures close the device.
    gc.collect()
    return True


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    on, why = _decide(session)
    if not on or session.testsfailed or session.config.option.collectonly:
        print(f"\nPRECOMPILE: off ({why})", flush=True)
        return None
    import numpy
    import torch
    import ttnn

    items = session.items
    present = _kernels_in_cache()
    # The dry pass draws from the random generators and may record properties on the items both passes share; the
    # real pass must start from what plain pytest would see.
    rng = (random.getstate(), numpy.random.get_state(), torch.get_rng_state())
    properties = [list(item.user_properties) for item in items]
    real_set_fabric_config = _open_fabric_without_routers(ttnn)
    stops = _swap_stops(None)
    start = time.monotonic()
    _STATE["dry"] = True
    try:
        for i, item in enumerate(items):
            # log=False: no report of a dry body reaches the terminal, JUnit or -x. nextitem=None on the last item
            # tears every fixture down, so the real pass sets everything up again.
            runtestprotocol(item, nextitem=items[i + 1] if i + 1 < len(items) else None, log=False)
    finally:
        _STATE["dry"] = False
        ttnn.set_fabric_config = real_set_fabric_config
        _swap_stops(stops)
    dry_seconds = time.monotonic() - start
    ttnn._ttnn.device.WaitForQueuedBuilds()
    wait_seconds = time.monotonic() - start - dry_seconds
    random.setstate(rng[0])
    numpy.random.set_state(rng[1])
    torch.set_rng_state(rng[2])
    for item, saved in zip(items, properties):
        item.user_properties[:] = saved
    _STATE["present"], _STATE["ahead"] = present, _kernels_in_cache() - present
    print(
        f"\nPRECOMPILE: on ({why}). Dry pass over {len(items)} tests took {dry_seconds:.1f} s "
        f"({_STATE['bodies']} bodies, {_STATE['ended_early']} ended early, {_STATE['timed_out']} timed out), "
        f"then {wait_seconds:.1f} s waiting for builds. Kernels: {present} already present, "
        f"{_STATE['ahead']} built ahead.",
        flush=True,
    )
    return None  # pytest's own loop now runs the real pass


def pytest_sessionfinish(session):
    if "ahead" in _STATE:
        late = _kernels_in_cache() - _STATE["present"] - _STATE["ahead"]
        print(f"\nPRECOMPILE: {late} kernels built late, during the real pass.", flush=True)
