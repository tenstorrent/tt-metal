# Round 3 eltwise binary: record a hash of every tensor ttnn.to_torch returns, per test, so that two runs (main and an
# opt-in) can be compared bit for bit. The hashes go to $EB_HASH_OUT (JSON) at the end of the session.
import hashlib
import json
import os

import pytest
import torch
import ttnn

_state = {"test": None, "hashes": {}, "outcome": {}}
_orig_to_torch = ttnn.to_torch


def _hashing_to_torch(*args, **kwargs):
    t = _orig_to_torch(*args, **kwargs)
    try:
        b = t.detach().cpu().contiguous()
        h = hashlib.sha1(b.view(torch.uint8).numpy().tobytes()).hexdigest()[:16] + f":{tuple(b.shape)}:{b.dtype}"
    except Exception as e:  # noqa: BLE001
        h = f"unhashable:{type(e).__name__}"
    _state["hashes"].setdefault(_state["test"], []).append(h)
    return t


ttnn.to_torch = _hashing_to_torch


def pytest_runtest_setup(item):
    _state["test"] = item.nodeid


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" or (rep.when == "setup" and rep.outcome != "passed"):
        _state["outcome"][item.nodeid] = rep.outcome


def pytest_sessionfinish(session):
    out = os.environ.get("EB_HASH_OUT")
    if out:
        json.dump({"hashes": _state["hashes"], "outcome": _state["outcome"]}, open(out, "w"), indent=0)
