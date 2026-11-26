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


def test_moe_rows_tile(device):
    """TILE-layout weights (the cache the chunked prefill needs): FusedMoeExperts runs R rows one by one; every row
    bit-equal to the 1-row call. Before, rows past 0 were not computed and a verify pass on a TILE cache gave
    tokens that differ from plain decode."""
    from models.demos.inkling2_mini.tt.fused.moe_expert import FusedMoeExperts
    from models.demos.inkling2_mini.tt.moe import FUSED_PLACEMENT

    g = torch.Generator().manual_seed(3)
    gu = torch.cat(
        [
            torch.randn(N, H, I, generator=g, dtype=torch.bfloat16) * (3 / math.sqrt(H)),
            torch.randn(N, H, I, generator=g, dtype=torch.bfloat16) * (8 / math.sqrt(H)),
        ],
        2,
    )
    dn = torch.randn(N, I, H, generator=g, dtype=torch.bfloat16) / math.sqrt(I)
    bad = []
    for name, (dg, dd) in DT.items():
        dev_w = [
            ttnn.from_torch(t.unsqueeze(0), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            for t, dt in ((gu, dg), (dn, dd))
        ]
        op = FusedMoeExperts(device, *dev_w, sticks=False, **FUSED_PLACEMENT)
        for R, overlap in ((2, 4), (3, 2), (3, 8)):
            bad += _case(device, op, R, overlap, f"tile-{name}-R{R}-o{overlap}")
        for t in dev_w:
            ttnn.deallocate(t)
    assert not bad, bad


def test_moe_rows_ring(device):
    """PF layout with the prefetcher ring (tt/mk/expert_pipe.py): the multi-row step reads the marked union experts
    from the ring, pops every slab and writes the next guess (row 0's top experts); rows bit-equal to 1-row DRAM
    calls. Masks: hits in row 0 only, in a later row only, a miss, hits + a miss."""
    from models.demos.inkling2_mini.tt.mk.expert_pipe import ExpertPipe

    experts = _experts(device)
    ops = {dt: experts[(dt, "pf")] for dt in DT}
    refs = {dt: BankMoeExperts(device, (v.w_gate_up, v.gu_x), (v.w_down, v.dn_x)) for dt, v in ops.items()}
    ep = ExpertPipe(device, list(ops.values()), 240 << 10)
    ep.start()
    bad = []

    def mask_t(sel):
        m = torch.zeros(1, 1, 1, N, dtype=torch.int32)
        m[..., list(sel)] = 1
        return ttnn.from_torch(
            m, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    try:
        mout = mask_t([])
        g = torch.Generator().manual_seed(9)
        for dt, op in ops.items():
            for R in (2, 3):
                ids = [torch.randperm(N, generator=g)[:K].tolist()]
                for _ in range(R - 1):
                    rest = [e for e in torch.randperm(N, generator=g).tolist() if e not in ids[-1]][: K - 3]
                    ids.append(rest + ids[-1][:3])
                union = {e for row in ids for e in row}
                miss = [e for e in range(N) if e not in union]
                w = torch.rand(R, K, generator=g) + 0.5
                x = torch.randn(R, H, generator=g)
                xt = _put(device, x.reshape(1, 1, R, H), ttnn.bfloat16)
                wt = _put(device, w.reshape(1, 1, R, K), ttnn.bfloat16)
                it = _put(device, torch.tensor(ids, dtype=torch.int32).reshape(1, 1, R, K), ttnn.uint16)
                ref = []
                for r in range(R):
                    y1 = refs[dt](
                        _put(device, x[r].reshape(1, 1, 1, H), ttnn.bfloat16),
                        _put(device, w[r].reshape(1, 1, 1, K), ttnn.bfloat16),
                        _put(device, torch.tensor(ids[r], dtype=torch.int32).reshape(1, 1, 1, K), ttnn.uint16),
                    )
                    ref.append(ttnn.to_torch(y1).reshape(-1, H)[0])
                want = sorted(
                    torch.tensor(ids[0])[torch.argsort(w[0].bfloat16().float(), descending=True)[:2]].tolist()
                )
                late = [e for e in ids[-1] if e not in ids[0]]
                cases = {"row0": ids[0][3:5], "late": late[:2], "miss": miss[:1], "mix": [ids[0][5], miss[1]]}
                for cn, sel in cases.items():
                    mt = mask_t(sel)
                    ttnn.synchronize_device(device)
                    ep.request(op, mt)
                    y = op(xt, wt, it, guess=mt, guess_out=mout, n_guess=2, signal=ep.signal)
                    ep.stepped()
                    got = ttnn.to_torch(y).reshape(R, H)
                    for r in range(R):
                        if not torch.equal(got[r].view(torch.int16), ref[r].view(torch.int16)):
                            bad.append((dt, R, cn, r))
                    m = sorted(torch.nonzero(ttnn.to_torch(mout).reshape(-1).int()).reshape(-1).tolist())
                    if m != want:
                        bad.append((dt, R, cn, "mask", m, want))
    finally:
        ep.close()
        ep.stop()
    assert not bad, bad
