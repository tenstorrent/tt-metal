# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (torch, no ttnn, no device) checks of the sequence-sharded DSA plan on the 8x4 Galaxy mesh
(doc/perf/shard_8x4_design.md; tt/shard_plan.py Geometry(8, 4)): the geometry, the stream / link / packet-header
plan of every chip, the two reduce-scatter rounds emulated against a direct 32-way combine, the global top-512 merge
over 32 candidate lists against the serial reference. The 2x4 plan is regression-checked the same way.

    pytest models/demos/glm53_flash/tests/test_shard_plan_8x4.py -q    (never beside a running mesh job: imports torch only)
"""

from __future__ import annotations

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import shard_plan as SP

D = 512


def _direct_parts(gen, n: int, keys_per_chip: int = 48):
    """per chip random local keys / values; parts_fn(rank, q_tile) = the chip's partial softmax stats over its keys for
    the 32 heads of q_tile; plus the exact attention over EVERY chip's keys per head (the reference)"""
    K = {r: torch.randn(keys_per_chip, D, generator=gen) / D**0.5 for r in range(n)}
    V = {r: torch.randn(keys_per_chip, D, generator=gen) for r in range(n)}

    def parts(r, qt):  # qt [32, D] -> (m [32], l [32], O [32, D]) over chip r's keys
        s = qt @ K[r].T  # [32, keys]
        m = s.max(dim=1).values
        e = torch.exp(s - m[:, None])
        return m, e.sum(dim=1), e @ V[r]

    def exact(q):  # q [64, D] -> o [64, D] over all chips' keys
        Kall = torch.cat([K[r] for r in range(n)])
        Vall = torch.cat([V[r] for r in range(n)])
        return torch.softmax(q @ Kall.T, dim=1) @ Vall

    return parts, exact


@pytest.mark.parametrize("R,C", [(2, 4), (8, 4)])
def test_geometry(R, C):
    g = SP.Geometry(R, C)
    n = R * C
    assert g.n == n and g.P * g.L == n and g.hpc * n == 64
    assert g.piece_rows * (n // 2) == 32
    assert sorted(g.tile_ranks(0) + g.tile_ranks(1)) == list(range(n))
    for r in range(n):
        assert g.rank(*g.coord(r)) == r
        assert r in g.tile_ranks(g.tile(r))
        assert g.piece(r) == g.tile_ranks(g.tile(r)).index(r)
        assert g.pass_tile(r, 1) == g.tile(r) and g.pass_tile(r, 0) == 1 - g.tile(r)
        assert len(g.column_peers(r)) == g.P - 1 and all(g.line_pos(s) == g.line_pos(r) for s in g.column_peers(r))
        assert len(g.line_peers(r)) == g.L - 1
    assert len({g.ag_blk(r) for r in range(n)}) == n  # distinct y_ag blocks
    if R == 8:
        assert g.P == 8 and g.rs1_rows == 8 and g.rs2_rows == 2 and g.piece_rows == 2
        assert g.tile(0) == 0 and g.tile(15) == 0 and g.tile(16) == 1 and g.tile(31) == 1
        # the final owner of heads [2 r, 2 r + 2): round 1 slice [8 i, +8) of row i, round 2 sub-slice [2 j, +2)
        for r in range(n):
            i, j = g.pair_pos(r), g.line_pos(r)
            assert g.rs1_rows * i + g.rs2_rows * j == g.hpc * r


@pytest.mark.parametrize("R,C", [(2, 4), (8, 4)])
@pytest.mark.parametrize("t", [1, 4, 16])
def test_streams_and_audits(R, C, t):
    g = SP.Geometry(R, C)
    st = g.streams(t)
    assert set(st) == {"ag1", "ag2", "rs1", "rs2"}
    assert st["ag2"]["chunks"] == g.P * g.ag_chunks(t)
    assert st["rs1"]["kind"] == ("rs2" if g.P == 2 else "scatter")
    for r in range(R):
        for c in range(C):
            SP.audit_links(g.holders(r, c, t))
            SP.audit_headers(g.header_budget(r, c, t))
            for name in st:
                b = g.block(name, t, r, c)
                n_axis = g.axis_len(st[name]["axis"])
                assert b["up"] + b["dn"] == n_axis - 1
                if st[name]["kind"] == "scatter":
                    assert sorted(q for q, _, _, _ in b["dsts"]) == [q for q in range(n_axis) if q != b["p"]]
                    assert all(h == abs(q - b["p"]) for q, _, _, h in b["dsts"])


@pytest.mark.parametrize("R,C", [(2, 4), (8, 4)])
def test_exchange_matches_direct(R, C):
    gen = torch.Generator().manual_seed(1234 + R)
    g = SP.Geometry(R, C)
    parts, exact = _direct_parts(gen, g.n)
    q = torch.randn(64, D, generator=gen)
    out = SP.emulate_exchange(g, q, parts)
    ref = exact(q)
    for r in range(g.n):
        want = ref[g.hpc * r : g.hpc * (r + 1)]
        assert out[r].shape == want.shape, (r, out[r].shape)
        torch.testing.assert_close(out[r], want, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("C,lanes", [(8, 1), (32, 1), (32, 2)])
def test_lane_merge_32_lists(C, lanes):
    gen = torch.Generator().manual_seed(7)
    n_pools = 32 * C * 6  # ~6 pool pages per chip
    keys = torch.randint(1, 1 << 31, (n_pools,), generator=gen, dtype=torch.int64)
    res = SP.emulate_lane_merge(keys, C, chunks=64, lanes=lanes, k=512)
    ref = SP.serial_merge(res["lists"], 0, k=512)
    assert res["T_g"] == ref["T_g"], (res["T_g"], ref["T_g"])
    assert res["core_quota"][0] == ref["core_quota"]
    assert SP.lanes_reasons(16, 2, 64, 4, 1, nchips=C) == [] if C <= 32 else True
    assert SP.lanes_reasons(32, 4, 64, 4, 1, nchips=32) == ["32 merge helpers among 16 lane chunk cores"]
