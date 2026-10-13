# pytest plugin (-p resbits, PYTHONPATH=ci_ab/plug): every result tensor the eltwise binary tests compare against golden is
# hashed bit for bit (raw bytes) and appended to $RESBITS_OUT with the node id, so two trees can be compared exactly.
import hashlib
import json
import os

import torch

_node = {"id": None}


def pytest_runtest_setup(item):
    _node["id"] = item.nodeid


def _wrap(mod):
    orig = getattr(mod, "passed_test", None)
    if orig is None or getattr(orig, "_resbits", False):
        return

    def passed_test(golden_tensor, res_tensor, *a, **k):
        try:
            b = res_tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            g = golden_tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            with open(os.environ["RESBITS_OUT"], "a") as fh:
                fh.write(json.dumps({"node": _node["id"], "res": hashlib.sha1(b).hexdigest(), "golden": hashlib.sha1(g).hexdigest(), "n": len(b)}) + "\n")
        except Exception as e:
            with open(os.environ["RESBITS_OUT"], "a") as fh:
                fh.write(json.dumps({"node": _node["id"], "error": str(e)}) + "\n")
        return orig(golden_tensor, res_tensor, *a, **k)

    passed_test._resbits = True
    mod.passed_test = passed_test


def pytest_collection_modifyitems(session, config, items):
    if not os.environ.get("RESBITS_OUT"):
        return
    import sys

    for name in ("test_eltwise_binary", "test_eltwise_binary_broadcast"):
        if name in sys.modules:
            _wrap(sys.modules[name])
