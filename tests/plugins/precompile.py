# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Up-front precompile for a ttnn pytest run.

Before the tests run for real, every test body runs once in a dry run (ttnn.device.BeginDryRun): nothing is sent to
the device, and each program the body would have launched has its kernel builds queued in the background. The real
pass then finds the kernels in Metal's kernel cache and its tests do not wait for them. Builds still queued when the
real pass starts go on behind it, in the order the tests need them.

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
_STATE = {"dry": False, "bodies": 0, "ended_early": 0, "timed_out": 0, "body_seconds": 0.0}


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


# Shared comparison helpers, by name prefix. A dry body reads no real data from the device, so a check on device data
# says nothing, and what follows the first check is mostly more reading and checking: the body ends there.
_STOP_AT_CHECKS = {
    "tt_py_test_utils_common.comparison_funcs": "comp_",
    "tt_py_test_utils_common.utils_for_testing": "assert_ comp_ check_",
}
# Random test data, and what a dry body gets instead: memory that is allocated but never written. The data is not sent
# to a device, and filling large tensors is a large part of a body's host work.
_UNFILLED = {"rand": "empty", "randn": "empty", "rand_like": "empty_like", "randn_like": "empty_like"}


def _stop(*args, **kwargs):
    raise _BodyStop()


def _unfilled(empty):
    def stand_in(*args, generator=None, **kwargs):
        return empty(*args, **kwargs)

    return stand_in


def _stand_ins():
    """{id(real function): stand-in} for the calls that end a dry body and for random test data."""
    import importlib

    import torch

    stand_ins = {id(getattr(torch, name)): _unfilled(getattr(torch, empty)) for name, empty in _UNFILLED.items()}
    for module_name, names in _STOP_AT.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in names.split():
            if hasattr(module, name):
                stand_ins[id(getattr(module, name))] = _stop
    for module_name, prefixes in _STOP_AT_CHECKS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name, value in vars(module).items():
            is_own = callable(value) and getattr(value, "__module__", "") == module.__name__
            if is_own and name.startswith(tuple(prefixes.split())):
                stand_ins[id(value)] = _stop
    return stand_ins


def _rebind(stand_ins):
    """Put the stand-ins in place of the real functions in every loaded module, also where a test file imported one
    by name. Returns what was replaced, for _restore."""
    import sys

    replaced = []
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            stand_in = stand_ins.get(id(value))
            if stand_in is not None:
                replaced.append((namespace, name, value))
                namespace[name] = stand_in
    return replaced


def _restore(replaced):
    for namespace, name, value in replaced:
        namespace[name] = value


def _asks_for_fabric(item):
    params = getattr(getattr(item, "callspec", None), "params", {}).get("device_params")
    config = params.get("fabric_config") if isinstance(params, dict) else None
    return config is not None and getattr(config, "name", "") != "DISABLED"


def _decide(session):
    """Whether this run gets a dry pass, and what was looked at. Decided before the dry pass, from this run alone."""
    items = session.items
    functions = {(item.path, getattr(item, "originalname", item.name)) for item in items}
    seen = (
        f"{len(items)} tests, {len(functions)} test functions, {sum(map(_asks_for_fabric, items))} asking for a "
        f"fabric config, {len(os.sched_getaffinity(0))} CPUs, {_kernels_in_cache()} kernels in the cache"
    )
    if _MODE in ("on", "off"):
        return _MODE == "on", f"PRECOMPILE={_MODE}; {seen}"
    # No rule yet: nothing a run can see before its dry pass separated the jobs that gain from the ones that lose.
    return False, f"auto, no rule decides on yet; {seen}"


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
    function. A call that names a manager mode other than DEFAULT is passed through unchanged.

    Not with the watcher on: it checks the kernel on every core, and the fabric cores of a mesh opened this way still
    hold what the previous run left there."""
    real = ttnn.set_fabric_config
    if os.environ.get("TT_METAL_WATCHER"):
        return real
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
    body_start = time.monotonic()
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
        _STATE["body_seconds"] += time.monotonic() - body_start
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        ttnn._ttnn.device.EndDryRun()
    # What the body left behind must be gone before its fixtures close the device. Its objects are young; a full
    # collection costs about 70 ms per test with torch and ttnn loaded.
    gc.collect(1)
    return True


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    start = time.monotonic()
    on, why = _decide(session)
    if not on or session.testsfailed or session.config.option.collectonly:
        print(f"\nPRECOMPILE: off ({why}). Deciding took {time.monotonic() - start:.2f} s.", flush=True)
        return None
    import numpy
    import torch
    import ttnn

    items = session.items
    present = _STATE["present"] = _kernels_in_cache()
    # The dry pass draws from the random generators and may record properties on the items both passes share; the
    # real pass must start from what plain pytest would see.
    rng = (random.getstate(), numpy.random.get_state(), torch.get_rng_state())
    properties = [list(item.user_properties) for item in items]
    real_set_fabric_config = _open_fabric_without_routers(ttnn)
    replaced = _rebind(_stand_ins())
    start = time.monotonic()
    _STATE["dry"] = True
    # Some fixtures collect garbage before every test (models/conftest.py). A full collection costs tens of
    # milliseconds with torch and ttnn loaded; in the dry pass the young generations are enough, as for the plugin's
    # own collection after a body. PRECOMPILE_FULL_GC=1 is a temporary switch for comparing the two.
    real_collect = gc.collect
    if not os.environ.get("PRECOMPILE_FULL_GC"):
        gc.collect = lambda generation=1: real_collect(min(generation, 1))
    try:
        ttnn._ttnn.device.SetDryPass(True)
        for i, item in enumerate(items):
            # log=False: no report of a dry body reaches the terminal, JUnit or -x. nextitem=None on the last item
            # tears every fixture down, so the real pass sets everything up again.
            runtestprotocol(item, nextitem=items[i + 1] if i + 1 < len(items) else None, log=False)
    finally:
        _STATE["dry"] = False
        gc.collect = real_collect
        ttnn._ttnn.device.SetDryPass(False)
        ttnn.set_fabric_config = real_set_fabric_config
        _restore(replaced)
    random.setstate(rng[0])
    numpy.random.set_state(rng[1])
    torch.set_rng_state(rng[2])
    for item, saved in zip(items, properties):
        item.user_properties[:] = saved
    print(
        f"\nPRECOMPILE: on ({why}). Dry pass over {len(items)} tests took {time.monotonic() - start:.1f} s "
        f"({_STATE['bodies']} bodies, {_STATE['ended_early']} ended early, {_STATE['timed_out']} timed out; "
        f"{_STATE['body_seconds']:.1f} s in bodies). Kernels: {present} already present, "
        f"{_kernels_in_cache() - present} built so far; queued builds go on behind the real pass.",
        flush=True,
    )
    return None  # pytest's own loop now runs the real pass


def pytest_sessionfinish(session):
    if "present" in _STATE:
        import ttnn

        # Nothing needs a kernel any more: builds still in the queue are dropped.
        ahead = ttnn._ttnn.device.FinishQueuedBuilds()
        late = _kernels_in_cache() - _STATE["present"] - ahead
        print(f"\nPRECOMPILE: {ahead} kernels built ahead, {late} built late by the test that needed them.", flush=True)
