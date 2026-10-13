# sfpu_stateful round 2 (SS3) bit compare: after every TestConfig.run() of a consumer run, save the raw result-buffer
# bytes as the packer wrote them to $SS3_BITS_DIR/<node>__<n>.bin and append the sha256 to $SS3_BITS_DIR/index.jsonl.
# Load with: PYTHONPATH=<this dir> SS3_BITS_DIR=<dir> pytest -p ss3_bits ...
import hashlib
import json
import os
import re

import pytest

_state = {"node": None, "n": 0}


def pytest_runtest_setup(item):
    _state["node"] = item.nodeid
    _state["n"] = 0


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    out = os.environ.get("SS3_BITS_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    from helpers.test_config import TestConfig

    orig = TestConfig.run

    def run(self, *args, **kwargs):
        outcome = orig(self, *args, **kwargs)
        node = _state["node"] or "unknown"
        rec = {"node": node, "n": _state["n"], "variant": self.variant_id[:16]}
        try:
            raw = self.variant_stimuli.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION)
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", node.split("::", 1)[-1])[:180]
            path = os.path.join(out, f"{name}__{_state['n']}.bin")
            with open(path, "wb") as fh:
                fh.write(raw)
            rec.update(sha256=hashlib.sha256(raw).hexdigest(), nbytes=len(raw), file=os.path.basename(path))
        except Exception as exc:  # recorded, never fatal
            rec["error"] = repr(exc)
        with open(os.path.join(out, "index.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        _state["n"] += 1
        return outcome

    TestConfig.run = run
