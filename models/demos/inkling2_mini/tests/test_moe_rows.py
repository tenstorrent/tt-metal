# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Multi-row bank expert step (speculative verify, tt/fused/moe_bank.py _call_rows): every output row
must be bit-equal to the 1-row call on that row alone. Bank and PF layout, BFP4 and mixed weights."""

import math

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.moe_bank import BankMoeExperts, bank_weights, pf_weights

H, I, K, N = 2304, 1152, 8, 128
DT = {"bfp4": (ttnn.bfloat4_b, ttnn.bfloat4_b), "mixed": (ttnn.bfloat4_b, ttnn.bfloat8_b)}


def _experts(device):
    g = torch.Generator().manual_seed(3)
    gu = torch.cat(
        [
            torch.randn(N, H, I, generator=g, dtype=torch.bfloat16) * (3 / math.sqrt(H)),
            torch.randn(N, H, I, generator=g, dtype=torch.bfloat16) * (8 / math.sqrt(H)),
        ],
        2,
    )
    dn = torch.randn(N, I, H, generator=g, dtype=torch.bfloat16) / math.sqrt(I)
    ops = {}
    for name, (dg, dd) in DT.items():
        ops[(name, "pf")] = BankMoeExperts(
            device, pf_weights(device, gu, dg, "experts.w_gate_up"), pf_weights(device, dn, dd, "experts.w_down")
        )
        ops[(name, "bank")] = BankMoeExperts(
            device, bank_weights(device, gu, dg, "experts.w_gate_up"), bank_weights(device, dn, dd, "experts.w_down")
        )
    return ops


def _put(device, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


def test_moe_rows(device):
    """All cases in one test: the weights go to the device once (the device fixture is per test)."""
    experts = _experts(device)
    bad = []
    for layout in ("pf", "bank"):
        for dt in DT:
            for R, overlap in ((2, 4), (3, 2), (3, 8), (4, 3)):
                bad += _case(device, experts[(dt, layout)], R, overlap, f"{layout}-{dt}-R{R}-o{overlap}")
    assert not bad, bad


def _case(device, op, R, overlap, tag):
    """overlap: experts row r shares with row r - 1 (8 = the same set in another order). Returns failing rows."""
    g = torch.Generator().manual_seed(R * 10 + overlap)
    ids, prev = [], None
    for _ in range(R):
        if prev is None:
            row = torch.randperm(N, generator=g)[:K].tolist()
        else:
            keep = prev[:overlap]
            rest = [e for e in torch.randperm(N, generator=g).tolist() if e not in prev][: K - overlap]
            row = keep + rest
            row = [row[i] for i in torch.randperm(K, generator=g).tolist()]
        ids.append(row)
        prev = row
    w = torch.rand(R, K, generator=g) + 0.5
    x = torch.randn(R, H, generator=g)
    y = op(
        _put(device, x.reshape(1, 1, R, H), ttnn.bfloat16),
        _put(device, w.reshape(1, 1, R, K), ttnn.bfloat16),
        _put(device, torch.tensor(ids, dtype=torch.int32).reshape(1, 1, R, K), ttnn.uint16),
    )
    got = ttnn.to_torch(y).reshape(R, H)
    bad = []
    for r in range(R):
        y1 = op(
            _put(device, x[r].reshape(1, 1, 1, H), ttnn.bfloat16),
            _put(device, w[r].reshape(1, 1, 1, K), ttnn.bfloat16),
            _put(device, torch.tensor(ids[r], dtype=torch.int32).reshape(1, 1, 1, K), ttnn.uint16),
        )
        ref = ttnn.to_torch(y1).reshape(-1, H)[0]
        if not torch.equal(got[r].view(torch.int16), ref.view(torch.int16)):
            bad.append((tag, r, float((got[r].float() - ref.float()).abs().max())))
    return bad
