# pytest plugin (-p topkbits, PYTHONPATH=<this dir>): seeds every RNG before each test and hashes, bit for bit, every L1
# read the stimuli layer makes (results, raw result bytes, buffer C), so two LLK trees can be compared exactly.
# Output: one JSON line per read to $TOPKBITS_OUT with the node id, the read's ordinal, address, length and sha1.
import hashlib
import json
import os
import random

_state = {"id": None, "n": 0}


def pytest_runtest_setup(item):
    _state["id"] = item.nodeid
    _state["n"] = 0
    random.seed(1234)
    try:
        import numpy as np

        np.random.seed(1234)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(1234)
    except Exception:
        pass


def pytest_configure(config):
    out = os.environ.get("TOPKBITS_OUT")
    if not out:
        return
    import helpers.stimuli_config as sc

    orig = sc.read_from_device
    if getattr(orig, "_topkbits", False):
        return

    def read_from_device(location, addr, *a, **k):
        data = orig(location, addr, *a, **k)
        try:
            b = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
            h = hashlib.sha1(b).hexdigest()
            n = len(b)
        except Exception as e:  # keep the test running; the line marks the failure
            h, n = f"error {e}", -1
        _state["n"] += 1
        with open(out, "a") as fh:
            fh.write(json.dumps({"node": _state["id"], "i": _state["n"], "addr": addr, "n": n, "sha1": h}) + "\n")
        return data

    read_from_device._topkbits = True
    sc.read_from_device = read_from_device
