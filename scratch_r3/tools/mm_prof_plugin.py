# Round 3 matmul: run each selected test function MM_REPS times in a row under the device profiler, a tracy signpost
# "<nodeid>#<k>" before each call, and read the device profiler after the last call, so the ops report splits the launches
# of every repetition. The first repetition is the warm-up (program cache, first-use effects); the reduce drops it.
import gc
import os

import pytest

N_REPS = int(os.environ.get("MM_REPS", "4"))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    import ttnn
    from tracy import signpost

    fn = pyfuncitem.obj
    args = {a: pyfuncitem.funcargs[a] for a in pyfuncitem._fixtureinfo.argnames}
    dev = args.get("device", args.get("mesh_device"))
    failure = None
    for k in range(N_REPS):
        if dev is not None:
            ttnn.synchronize_device(dev)
        signpost(header=f"{pyfuncitem.nodeid}#{k}")
        try:
            fn(**args)
        except AssertionError as e:  # a numeric check: the launches ran, keep measuring and report it at the end
            failure = failure or e
        gc.collect()
    if dev is not None:
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
    if failure is not None:
        raise failure
    return True


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # MM_SELECT: comma-separated node id substrings; tracy re-joins argv through /bin/sh, so -k with spaces breaks
    sel = [k for k in os.environ.get("MM_SELECT", "").split(",") if k]
    if not sel:
        return
    keep = [i for i in items if any(k in i.nodeid for k in sel)]
    config.hook.pytest_deselected(items=[i for i in items if i not in keep])
    items[:] = keep
