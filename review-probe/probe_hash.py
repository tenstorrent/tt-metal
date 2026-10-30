"""pytest plugin (local review probe): record a sha256 of every result tensor handed to passed_test.

Use: PROBE_HASH_OUT=out.tsv PYTHONPATH=<tools dir> pytest -p probe_hash ...
Writes nodeid, result hash, golden hash, dtype, numel per call; compare two trees' files to check
bit-identity (matching golden hashes show both runs used the same stimuli).
"""
import hashlib
import os

import torch

OUT = os.environ.get("PROBE_HASH_OUT", "probe_hash.tsv")


def _sha(t):
    t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    b = t.detach().cpu().contiguous()
    raw = b.numpy().tobytes() if b.dtype == torch.bool else b.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16], b


def _wrap(orig, nodeid):
    def wrapped(golden, res, *a, **kw):
        h, b = _sha(res)
        gh, _ = _sha(golden)
        with open(OUT, "a") as f:
            f.write(f"{nodeid}\t{h}\t{gh}\t{b.dtype}\t{b.numel()}\n")
        return orig(golden, res, *a, **kw)

    return wrapped


def pytest_runtest_setup(item):
    mod = getattr(item, "module", None)
    if mod is not None and hasattr(mod, "passed_test"):
        orig = getattr(mod, "_probe_orig_passed_test", None) or mod.passed_test
        mod._probe_orig_passed_test = orig
        mod.passed_test = _wrap(orig, item.nodeid)
