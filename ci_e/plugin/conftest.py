# Scratch, not for merge. Loaded by passing this directory as an extra pytest argument (it holds no tests).
#   SEL_FILE=<file>  keep only the node ids listed (an entry without "[" keeps every parametrization of that function);
#                    ids the tree does not have are dropped, not errors
#   BITS_TAG=<tag>   print "BITS <tag> <node> #<i> <shape> <dtype> <sha256>" for every ttnn.to_torch result of a test
import hashlib
import os

import pytest

_cur = {"node": None, "i": 0}
_TAG = os.environ.get("BITS_TAG")


def _install():
    import torch
    import ttnn

    orig = ttnn.to_torch

    def wrapped(*a, **k):
        t = orig(*a, **k)
        if _cur["node"] is not None and isinstance(t, torch.Tensor):
            x = t.detach().contiguous().cpu()
            iv = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(x.element_size())
            raw = (x.view(iv) if x.is_floating_point() and iv else x).numpy().tobytes()
            print(f"BITS {_TAG} {_cur['node']} #{_cur['i']} {tuple(x.shape)} {x.dtype} {hashlib.sha256(raw).hexdigest()[:24]}",
                  flush=True)
            _cur["i"] += 1
        return t

    ttnn.to_torch = wrapped


if _TAG:
    _install()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    sel = os.environ.get("SEL_FILE")
    if not sel:
        return
    wanted = {l.strip() for l in open(sel) if l.strip()}
    keep = [it for it in items if it.nodeid in wanted or it.nodeid.split("[")[0] in wanted]
    dropped = [it for it in items if it not in keep]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep
    print(f"SEL kept {len(keep)} of {len(wanted)} listed", flush=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _cur["node"], _cur["i"] = item.nodeid.split("::")[-1], 0
    yield
    _cur["node"] = None
