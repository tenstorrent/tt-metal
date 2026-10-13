# Datacopy fold A/B (scratch, not for merge): seeds every test from its node id and writes the raw bytes of each result it
# collects to $BITDUMP_DIR/<hash>__<n>.bin, with <hash> -> node id in $BITDUMP_DIR/index.tsv. Load with -p bitdump_plugin.
import hashlib
import os
import random

import numpy as np
import pytest
import torch
from helpers.stimuli_config import StimuliConfig

OUT = os.environ.get("BITDUMP_DIR", "/tmp/bitdump")
_STATE = {"name": None, "n": 0}
_real_collect = StimuliConfig.collect_results


def _collect(self, location="0,0"):
    raw = self.collect_raw_result_bytes(location)
    with open(os.path.join(OUT, f"{_STATE['name']}__{_STATE['n']}.bin"), "wb") as f:
        f.write(raw)
    _STATE["n"] += 1
    return _real_collect(self, location)


@pytest.fixture(autouse=True)
def _bitdump(request, monkeypatch):
    os.makedirs(OUT, exist_ok=True)
    h = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:20]
    _STATE["name"], _STATE["n"] = h, 0
    with open(os.path.join(OUT, "index.tsv"), "a") as f:
        f.write(f"{h}\t{request.node.nodeid}\n")
    seed = int(h[:8], 16)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    monkeypatch.setattr(StimuliConfig, "collect_results", _collect)
    yield
