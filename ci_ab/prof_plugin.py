"""pytest plugin: after each test call, read the device profiler and append per-program DEVICE KERNEL DURATION [ns] to $PROF_OUT (jsonl).
Use: PROF_OUT=<file> pytest -p prof_plugin <ids> (PYTHONPATH must contain this dir). Test must leave the device open until after the call (device fixture)."""
import json, os, hashlib, pytest, random
_BITS = []
def _install():
    import ttnn, torch
    orig = ttnn.to_torch
    def wrapped(*a, **k):
        t = orig(*a, **k)
        try:
            x = t.detach().cpu()
            x = x.to(torch.float32) if x.is_floating_point() else x
            _BITS.append(hashlib.md5(x.contiguous().numpy().tobytes()).hexdigest())
        except Exception as e:
            _BITS.append("err:" + repr(e)[:40])
        return t
    ttnn.to_torch = wrapped
_installed = False
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _installed
    import torch
    if not _installed:
        _install(); _installed = True
    _BITS.clear(); torch.manual_seed(1234); random.seed(1234)
    try:
        import numpy; numpy.random.seed(1234)
    except Exception: pass
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    dev = item.funcargs.get("device") or item.funcargs.get("mesh_device")
    rec = {"id": item.nodeid, "bits": hashlib.md5("".join(_BITS).encode()).hexdigest() if _BITS else None, "nbits": len(_BITS), "passed": outcome.excinfo is None, "skip": outcome.excinfo is not None and "Skipped" in repr(outcome.excinfo[1]), "progs": {}}
    try:
        import ttnn
        ttnn.synchronize_device(dev); ttnn.ReadDeviceProfiler(dev)
        d = ttnn.get_latest_programs_perf_data()
        for key, prog in (d.items() if isinstance(d, dict) else enumerate(d)):
            for r in (prog if isinstance(prog, list) else [prog]):
                v = getattr(r, "program_analyses_results", None) or {}
                x = v.get("DEVICE KERNEL DURATION [ns]") if isinstance(v, dict) else None
                if x is not None:
                    rec["progs"].setdefault(str(key), []).append(x.duration if hasattr(x, "duration") else x)
        if not rec["progs"]:
            rec["diag"] = {"type": type(d).__name__, "len": len(d) if hasattr(d, "__len__") else None}
            try:
                k0 = next(iter(d)) if isinstance(d, dict) else 0; p0 = d[k0]; r0 = p0[0] if isinstance(p0, list) else p0
                rec["diag"].update(key=str(k0)[:80], rtype=type(r0).__name__, attrs=[a for a in dir(r0) if not a.startswith("_")][:30],
                                   par=str(getattr(r0, "program_analyses_results", None))[:300])
            except Exception as e2: rec["diag"]["e2"] = repr(e2)[:120]
    except Exception as e:
        rec["err"] = repr(e)[:200]
    with open(os.environ["PROF_OUT"], "a") as f: f.write(json.dumps(rec) + "\n")

@pytest.fixture
def _pc_rep(request):
    return request.param
def pytest_generate_tests(metafunc):
    n = int(os.environ.get("REPEAT", "1"))
    if n > 1:
        metafunc.fixturenames.append("_pc_rep")
        metafunc.parametrize("_pc_rep", range(n), indirect=True, ids=lambda v: "rep%d" % v)
def pytest_collection_modifyitems(config, items):
    only = os.environ.get("PLUGIN_ONLY")
    if only:
        keys = [k for k in only.split("||") if k]
        keep = [it for it in items if any(all(p in it.nodeid for p in k.split("&&")) for k in keys)]
        config.hook.pytest_deselected(items=[it for it in items if it not in keep]); items[:] = keep
    ch = os.environ.get('CHUNK')
    if ch:
        k, n = [int(x) for x in ch.split('/')]; sz = -(-len(items) // n); keep = items[k * sz:(k + 1) * sz]
        config.hook.pytest_deselected(items=[it for it in items if it not in keep]); items[:] = keep
