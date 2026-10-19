"""Scratch pytest plugin (not for merge): after every TestConfig.run, append "<node id> <n> <sha1 of the result values>" to
$BITDUMP, so two trees' outputs can be compared value for value. Load with -p bitdump (ci_ab/sweep on PYTHONPATH)."""
import hashlib
import os
import struct

from helpers import test_config

_orig = test_config.TestConfig.run


def _run(self, *a, **k):
    out = _orig(self, *a, **k)
    res = getattr(out, "result", None)
    if res is not None and os.environ.get("BITDUMP"):
        h = hashlib.sha1()
        for v in res:
            h.update(struct.pack("<d", float(v)))
        tid = os.environ.get("PYTEST_CURRENT_TEST", "?").rsplit(" (", 1)[0]
        with open(os.environ["BITDUMP"], "a") as f:
            f.write(f"{tid}\t{len(res)}\t{h.hexdigest()[:16]}\n")
    return out


test_config.TestConfig.run = _run
