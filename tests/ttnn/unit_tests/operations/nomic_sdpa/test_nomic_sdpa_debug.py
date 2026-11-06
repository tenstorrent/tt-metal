# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 debug: which (b, h, query tile-row) blocks go wrong under the K/V pair multicast, per knob setting.

DO NOT DELETE — documents the 16x256 corruption hunt (3 segments a core, kv_depth = 2 ring reuse).
Each bad tile-row is mapped to its core (core_row_start) and that core's multicast roles are printed.
"""

import importlib

import pytest
import torch
import ttnn

from eval.golden_tests.nomic_sdpa.helpers import make_qkv, reference
from ttnn.operations.nomic_sdpa import nomic_sdpa

op = importlib.import_module("ttnn.operations.nomic_sdpa.nomic_sdpa")
desc = importlib.import_module("ttnn.operations.nomic_sdpa.nomic_sdpa_program_descriptor")


@pytest.mark.parametrize("knobs", [{"KV_MCAST": 0, "KV_DEPTH_L1": 1}, {"KV_MCAST": 0}, {"KV_MCAST": 1}], ids=str)
@pytest.mark.parametrize("shape", [(16, 256), (32, 128), (4, 512)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_bad_rows(device, shape, knobs, monkeypatch):
    for name, value in knobs.items():
        monkeypatch.setattr(op, name, value)
    monkeypatch.setattr(op, "_CALL_CACHE", {})
    batch, seqlen = shape
    q, k, v = make_qkv((batch, 12, seqlen, 64), "real", seed=0)
    mem = ttnn.L1_MEMORY_CONFIG
    tq, tk, tv = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        for t in (q, k, v)
    )
    got = ttnn.to_torch(nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=mem)).float().reshape(batch, seqlen, 12, 64)
    want = reference(q, k, v, 1.0).float().reshape(batch, seqlen, 12, 64)
    p = op.plan(batch, seqlen)
    s_t = p.seq_tiles
    err = (got - want).abs().reshape(batch, s_t, 32, 12, 64).amax(dim=(2, 4))  # (b, st, h)
    bad = []
    for b in range(batch):
        for h in range(12):
            for st in range(s_t):
                if err[b, st, h] > 0.3:
                    g = (b * 12 + h) * s_t + st
                    core = next(c for c in range(p.num_cores) if p.core_row_start[c] <= g < p.core_row_start[c + 1])
                    bad.append((g, core))

    class _Dev:  # roles only (coordinates irrelevant here)
        def worker_core_from_logical_core(self, c):
            return c

    roles, _ = desc.kv_multicast_topology(p, _Dev())
    cores = sorted({c for _, c in bad})
    for c in cores[:12]:
        rows = [g for g, cc in bad if cc == c]
        print(
            f"core {c}: rows {p.core_row_start[c]}..{p.core_row_start[c+1]-1} pairs {p.core_pairs(c)} roles {roles[c]}"
            f" bad rows {rows}"
        )
    assert not bad, f"{len(bad)} bad tile-rows on {len(cores)} cores (kv_depth={p.kv_depth})"
