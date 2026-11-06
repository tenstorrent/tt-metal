# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_PLE_POPULATE: populate_mmaps is idempotent, changes no bytes, and the call site obeys the flag (no device)."""

import glob
import os
from pathlib import Path
from unittest import mock

import pytest

HF = os.environ.get("HF_MODEL", "/home/ttuser/models/Qwen3.8-Flash-Next")
CACHE = os.environ.get("TT_CACHE_PATH", "/home/ttuser/models/qwen38_cache")
pytestmark = pytest.mark.skipif(not Path(HF).exists(), reason="Qwen3.8-Flash-Next checkpoint not present")

LANES, CALLS = 32, 6


def _lookup():
    from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
    from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38HostPLEEmbedding
    from models.demos.blackhole.qwen38_flash_next.ttnn import ple as P

    return P.Qwen38ResidentPLELookup(Qwen38HostPLEEmbedding(Qwen38Checkpoint(HF)))


def _ids():
    files = glob.glob(os.path.join(CACHE, "demo-context", "*.txt"))
    if files:
        from transformers import AutoTokenizer

        ids = AutoTokenizer.from_pretrained(HF)(open(files[0]).read()[:200000], add_special_tokens=False)["input_ids"]
    else:
        ids = [(i * 7919) % 200000 for i in range(20000)]
    return ids


def _run(lk, ids):
    stride = (len(ids) - CALLS - 8) // LANES
    offs = [i * stride for i in range(LANES)]
    ctxs = [None] * LANES
    out = []
    for s in range(CALLS):
        payload, ctxs = lk.lookup_lanes([ids[o + s] for o in offs], ctxs)
        out.append(bytes(payload))
    return out


def test_populate_idempotent_and_bytes_equal():
    lk = _lookup()
    ids = _ids()
    before = _run(lk, ids)
    first = lk.populate_mmaps()
    assert first > 0
    assert lk.populate_mmaps() == 0.0
    assert _run(lk, ids) == before


def test_flag_off_does_not_call_populate(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.ttnn import ple as P

    monkeypatch.setattr(P, "PLE_POPULATE", False)
    host = mock.MagicMock()
    with mock.patch.object(P, "Qwen38ResidentPLELookup") as cls:
        ple = P.Qwen38TTNNPLE.__new__(P.Qwen38TTNNPLE)
        ple.host_embedding = host
        ple._resident_lookup = None
        assert ple.resident_lookup is cls.return_value
        cls.return_value.populate_mmaps.assert_not_called()
    monkeypatch.setattr(P, "PLE_POPULATE", True)
    with mock.patch.object(P, "Qwen38ResidentPLELookup") as cls:
        ple = P.Qwen38TTNNPLE.__new__(P.Qwen38TTNNPLE)
        ple.host_embedding = host
        ple._resident_lookup = None
        ple.resident_lookup
        ple.resident_lookup
        cls.return_value.populate_mmaps.assert_called_once()
