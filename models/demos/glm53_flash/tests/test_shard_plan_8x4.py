# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (torch, no ttnn, no device) checks of the sequence-sharded DSA plan on the 8x4 Galaxy mesh
(doc/perf/shard_8x4_design.md; tt/shard_plan.py Geometry(8, 4)): the geometry, the stream / link / packet-header
plan of every chip, the two reduce-scatter rounds emulated against a direct 32-way combine, the global top-512 merge
over 32 candidate lists against the serial reference. The 2x4 plan is regression-checked the same way.

    pytest models/demos/glm53_flash/tests/test_shard_plan_8x4.py -q    (never beside a running mesh job: imports torch only)
"""

from __future__ import annotations

import contextlib
import random
import re
import sys

import pytest
import torch
from loguru import logger

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
    """the helpers' merge over the chips' sorted local top-512 lists == the serial merge; the kernels' local select
    (shard_plan.local_select over each chip's lane chunk cores) gives those lists bitwise in 4-bit digits and in R8's
    8-bit digits (GLM53_DSA_SHARD_R8), and the merge in 8-bit digits the same T_g / quotas"""
    gen = torch.Generator().manual_seed(7)
    n_pools = 32 * C * 6  # ~6 pool pages per chip
    keys = torch.randint(1, 1 << 31, (n_pools,), generator=gen, dtype=torch.int64)
    res = SP.emulate_lane_merge(keys, C, chunks=64, lanes=lanes, k=512)
    ref = SP.serial_merge(res["lists"], 0, k=512)
    assert res["T_g"] == ref["T_g"], (res["T_g"], ref["T_g"])
    assert res["core_quota"][0] == ref["core_quota"]
    for bits in (4, 8):
        got = SP.emulate_lane_merge(keys, C, chunks=64, lanes=lanes, k=512, digit_bits=bits)
        assert got == res, (C, lanes, bits)
    assert SP.lanes_reasons(16, 2, 64, 4, 1, nchips=C) == []
    assert SP.lanes_reasons(32, 4, 64, 4, 1, nchips=32) == ["32 merge helpers among 16 lane chunk cores"]


@pytest.mark.parametrize("lanes", [1, 4, 8])
def test_local_all_matches_the_rounds(lanes):
    """rev 72 GLM53_DSA_SHARD_LOCAL_ALL: on a chip lane with <= 512 closed local pools the skip (local_select_all: no
    rounds, no tie message, every key kept, the analytic offsets of dsa_shard.hpp local_all_offset) gives the radix
    select's kept pairs, records, record offsets and header count bitwise, in 4-bit and R8 digits, for random keys and
    for heavy ties (key 0 included); and the global merge over 32 chips around the 64 K boundary (some chips over 512
    local pools: the rounds, the others the skip) is the all-rounds merge"""
    gen = torch.Generator().manual_seed(72 + lanes)
    rnd = random.Random(72 + lanes)
    cl = 64 // lanes
    sizes = list(range(0, 40)) + list(range(480, 513)) + rnd.sample(range(40, 480), 24)
    for n in sizes:
        lq = torch.arange(n)
        for heavy in (False, True):
            hi = 4 if heavy else 1 << 32
            keys = torch.randint(0, hi, (n,), generator=gen, dtype=torch.int64)
            core_keys = SP.lane_core_keys(keys, lq, 64, lanes)
            assert [len(v) for v in core_keys] == [SP.local_all_pools(n, c, cl) for c in range(cl)], (n, lanes)
            got = SP.local_select_all(core_keys, 512)
            assert got["offsets"] == [SP.local_all_offset(n, c, cl) for c in range(cl)]
            assert got["messages"] == 0 and got["n_rec"] == n
            for bits in (4, 8):
                want = SP.local_select(core_keys, 512, bits)
                assert got["kept"] == want["kept"] and got["records"] == want["records"], (n, lanes, bits, heavy)
                assert got["n_rec"] == want["n_rec"] and got["offsets"] == [o for _, o in want["qpair"]], (n, bits)
                assert SP.x_ag_chunk(got["records"], n_rec=got["n_rec"]) == SP.x_ag_chunk(
                    want["records"], n_rec=want["n_rec"]
                )
    with expect_error(ValueError, "513 keys > k = 512"):
        SP.local_select_all([[1] * 513], 512)
    # the boundary: p = 65535 -> every chip at 512 (the skip everywhere); 66000 -> chips 0..3 over 512 (the rounds)
    assert {SP.n_closed_local(65535, r, 32) for r in range(32)} == {512}
    over = [r for r in range(32) if SP.n_closed_local(66000, r, 32) > 512]
    assert over == [0, 1, 2, 3], over
    for p in (65535, 66000):
        n_pools = (p + 1) // 4
        keys = torch.randint(1, 1 << 31, (n_pools,), generator=gen, dtype=torch.int64)
        g = torch.arange(n_pools)
        owner = (g // SP.POOL_PAGE) % 32
        rounds, mixed = [], []
        for r in range(32):
            mine = torch.nonzero(owner == r).reshape(-1)
            lq = (mine // SP.POOL_PAGE // 32) * SP.POOL_PAGE + mine % SP.POOL_PAGE
            ck = SP.lane_core_keys(keys[mine], lq, 64, lanes)
            sel = SP.local_select(ck, 512, 8)["records"]
            rounds.append(sel)
            n = SP.n_closed_local(p, r, 32)
            mixed.append(SP.local_select_all(ck, 512)["records"] if n <= 512 else sel)
            assert len(mixed[-1]) == SP.agk_count(p, r, 32)
        assert mixed == rounds
        for rank in (0, 5, 31):
            assert SP.helper_merge(mixed, rank, 512, digit_bits=8, kph=4) == SP.helper_merge(
                rounds, rank, 512, digit_bits=8, kph=4
            )


def test_local_all_knobs(monkeypatch):
    """GLM53_DSA_SHARD_LOCAL_ALL / GLM53_DSA_SHARD_ROUND_MARKS: default off, the 8x4 only, 0 / 1 only; ROUND_MARKS needs
    R8; the kernels and mk_mla.py carry the same define names (a renamed define would silently build knob-off)"""
    for k in ("GLM53_DSA_SHARD_LOCAL_ALL", "GLM53_DSA_SHARD_ROUND_MARKS"):
        monkeypatch.delenv(k, raising=False)
    assert not SP.shard_local_all(8) and not SP.shard_round_marks(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_LOCAL_ALL", "1")
    assert SP.shard_local_all(8) and not SP.shard_local_all(2)
    monkeypatch.setenv("GLM53_DSA_SHARD_ROUND_MARKS", "1")
    assert SP.shard_round_marks(8, r8=True) and not SP.shard_round_marks(2, r8=False)
    with expect_error(ValueError, "needs GLM53_DSA_SHARD_R8=1"):
        SP.shard_round_marks(8, r8=False)
    monkeypatch.setenv("GLM53_DSA_SHARD_LOCAL_ALL", "yes")
    with expect_error(ValueError, "GLM53_DSA_SHARD_LOCAL_ALL=yes: 0 or 1"):
        SP.shard_local_all(8)
    import os

    kd = os.path.join(SP.KDIR, "mla")
    for f, names in (
        ("ix_dm0.cpp", ("MLA_DSA_SHARD_LOCAL_ALL", "MLA_DSA_ROUND_MARKS")),
        ("att_dsa_shard_dm1.cpp", ("MLA_DSA_SHARD_LOCAL_ALL",)),
        ("dsa_shard.hpp", ("MLA_DSA_SHARD_LOCAL_ALL", "MLA_DSA_ROUND_MARKS", "local_all_offset")),
    ):
        with open(os.path.join(kd, f)) as fh:
            src = fh.read()
        assert all(nm in src for nm in names), (f, names)
    with open(os.path.join(os.path.dirname(SP.KDIR), "mk_mla.py")) as fh:
        src = fh.read()
    assert '("MLA_DSA_SHARD_LOCAL_ALL", "1")' in src and '("MLA_DSA_ROUND_MARKS", "1")' in src


def test_decode_probe_round_marks():
    """rev 72 ROUND_MARKS: ix_dm0's row with 16 marks (start, i1_in, y_in, 4 R8 rounds + the tie x (histograms in,
    broadcast), local_coord, ag_landed, merge_done) decodes to the 6 IX0 marks + the rounds and their spans; 6 marks
    (the knob off, or the lane's first token took LOCAL_ALL) decode as before"""
    p = torch.zeros(SP.PROBE_LANE_ROW["rs2"] + 1, 16, dtype=torch.int64)
    nc = [1000 + 1350 * k for k in range(9)]  # the root's 9 attention marks: att_start = 1000 cycles
    p[SP.PROBE_ROOT_NC, : len(nc)] = torch.tensor(nc)
    ix = [100 + 135 * k for k in range(16)]
    p[SP.PROBE_IX0, :16] = torch.tensor(ix)
    out = SP.decode_probe(p, line=4)
    us = lambda v: round((v - 1000) / 1350.0, 2)  # noqa: E731
    assert list(out["ix0"].values()) == [us(v) for v in ix[:3] + ix[13:]]
    assert [(x["hist_in"], x["out"]) for x in out["rounds"]] == [
        (us(ix[3 + 2 * r]), us(ix[4 + 2 * r])) for r in range(5)
    ]
    sp = out["span"]
    assert sp["r0_owners"] == round(us(ix[3]) - us(ix[2]), 2) and sp["tie_coord"] == round(us(ix[12]) - us(ix[11]), 2)
    assert sp["records"] == round(us(ix[13]) - us(ix[12]), 2)
    p[SP.PROBE_IX0] = 0
    p[SP.PROBE_IX0, :6] = torch.tensor(ix[:6])
    out = SP.decode_probe(p, line=4)
    assert "rounds" not in out and list(out["ix0"].values()) == [us(v) for v in ix[:6]]


def test_8x4_byte_formats_round_trip():
    """The 8x4 compact formats mirrored from kernels/mla/dsa_shard.hpp: the 2-row q piece (lat_dm1.cpp's write ->
    ix_dm1.cpp's 16-piece assembly = the whole tile, one page of 16 x 128 B), the round-1 8-row and round-2 2-row slice
    ship -> land round trips (att_dsa_shard_dm1.cpp), the page / offset helpers and the slice chunk layouts."""
    geo = SP.Geometry(8, 4)
    assert (geo.piece_rows, geo.rs1_rows, geo.rs2_rows) == (2, 8, 2)
    assert SP.piece_bytes(2) == 128 and SP.pieces_per_page(2) == 16  # "128 B x 16 tiles = ONE page"
    assert SP.piece_bytes(8) == 512 and SP.pieces_per_page(8) == 4
    assert SP.ag_pages(geo) == (SP.AG32_PPC, SP.AG32_QPAGES) == (6, 1)
    T = 4
    for d in range(16):
        assert SP.agx_q_page(geo, T, 1, 2, d) == ((1 * T + 2) * 6 + d // 16, (d % 16) * 128)
    assert SP.agx_q_page(geo, T, 0, 0, 16) == (1, 0)  # a 17th tile would start the next page
    assert SP.agx_cand_page(geo, T, 1, 2, 5000) == ((1 * T + 2) * 6 + 1 + 5000 // SP.TB, 5000 % SP.TB)
    assert SP.rs1_page(T, 3, 1) == (3 * T + 1) * SP.RS2_PPC and SP.rs2b_page(T, 3, 1) == (3 * T + 1) * SP.RS2B_PPC
    # the 2-row q pieces of 16 chips assemble the tile: chip j's rows 0..1 land at rows 2 j, 2 j + 1
    chips = [torch.arange(1024, dtype=torch.int32) + 10000 * j for j in range(16)]
    pieces = [SP.q_piece_rows(SP.tile_to_faces(c.reshape(32, 32)), 2) for c in chips]
    assert all(p.numel() == 64 for p in pieces)  # 128 B of bf16
    got = SP.faces_to_tile(SP.assemble_q_tile_rows(pieces, 2))
    want = torch.cat([chips[j].reshape(32, 32)[0:2] for j in range(16)])
    assert torch.equal(got, want)
    # the slice ship -> land round trips: 8-row (round 1) and 2-row (round 2) slices of a face-major tile
    for rows in (8, 2):
        src = torch.randn(32, 32)
        faces = SP.tile_to_faces(src)
        dst = torch.zeros(1024)
        for j in range(32 // rows):
            runs = SP.slice_runs_rows(faces, rows, j)
            assert runs[0].numel() == runs[1].numel() == SP.slice_run(rows, 1024)
            SP.land_slice_rows(dst, rows, j, runs)
        assert torch.equal(SP.faces_to_tile(dst), src)
    # the slice chunk layouts fit their pages: round 2 = 16 x 256 B + 2 x 256 B with fp32 O tiles (3 pages + pad)
    lay = SP.rs_layout(2, otb=4096)
    assert lay[-1][2] + 4096 * 2 // 32 == 16 * 256 + 2 * 256 <= SP.RS2B_PPC * SP.TB
    assert SP.rs2_layout(4096) == SP.rs_layout(8, 4096)


# ------------------------------------------------------------------ rev 64: KPH chips per merge helper (the 8x4)
def _lists(gen, case: str, n: int = 32) -> list:
    """32 chips' candidate lists [(key, core)] in (core, local) order (at most 512 records each), per case"""
    out = []
    for q in range(n):
        if case == "random":
            m = 512
        elif case == "ties":
            m = int(torch.randint(300, 513, (1,), generator=gen))
        elif case == "few":  # fewer than 512 candidates in the union: done after round 0, T_g = 0
            m = 15
        elif case == "empty":  # every other chip without candidates
            m = 0 if q % 2 else 512
        elif case == "one_chip":
            m = 512
        elif case == "all_empty":
            m = 0
        elif case == "lengths":  # R8 fuzz: every list length 0 .. 512
            m = int(torch.randint(0, 513, (1,), generator=gen))
        elif case in ("dups", "adversarial"):
            m = int(torch.randint(0, 513, (1,), generator=gen)) if q % 3 else 512
        elif case == "boundary":  # 513 candidates in the union: round 0 is not done, T_g = the smallest but one
            m = 32 if q < 16 else (1 if q == 16 else 0)
        else:
            raise ValueError(case)
        cores = torch.sort(torch.randint(0, 32, (m,), generator=gen)).values
        if case == "ties":
            keys = torch.randint(1, 40, (m,), generator=gen)
        elif case == "lengths":  # the full key range (positive and negative scores)
            keys = torch.randint(1, 1 << 32, (m,), generator=gen, dtype=torch.int64)
        elif case == "dups":  # 7 values on the 8-bit digits' edges: every level ties
            vals = torch.tensor([1, 0xFF, 0x100, 0x7FFFFFFF, 0x80000000, 0x800000FF, 0xFFFFFFFF])
            keys = vals[torch.randint(0, len(vals), (m,), generator=gen)]
        elif case == "adversarial":  # one shared top 24 bits (the coarse / fine picks of rounds 0-2 see one bin), the
            # low byte random or 0x00 / 0xFF / 0x0F / 0xF0 (the nibble edges of round 3)
            low = torch.randint(0, 256, (m,), generator=gen)
            edge = torch.tensor([0x00, 0xFF, 0x0F, 0xF0])[torch.randint(0, 4, (m,), generator=gen)]
            low = torch.where(torch.randint(0, 2, (m,), generator=gen) == 1, low, edge)
            keys = 0xBF8A5C00 + low + (low == 0).to(torch.int64) * (q % 2)
        elif case == "one_chip":  # chip 7 holds every winner
            keys = (
                torch.randint(1 << 30, 1 << 31, (m,), generator=gen)
                if q == 7
                else torch.randint(1, 1 << 20, (m,), generator=gen)
            )
        else:
            keys = torch.randint(1, 1 << 31, (m,), generator=gen)
        out.append([(int(k), int(c)) for k, c in zip(keys.tolist(), cores.tolist())])
    return out


def _topk_records(cores: list, k: int) -> list:
    """the reference local select: the k largest keys of the cores' lists, ties in (core, local) order, as records
    [(key, core)] in (core, local) order"""
    allk = [(kk, c, i) for c, v in enumerate(cores) for i, kk in enumerate(v)]
    sel = allk if len(allk) <= k else sorted(allk, key=lambda x: -x[0])[:k]
    return [(kk, c) for kk, c, _ in sorted(sel, key=lambda x: (x[1], x[2]))]


@pytest.mark.parametrize(
    "case", ["random", "ties", "few", "empty", "one_chip", "all_empty", "lengths", "dups", "adversarial", "boundary"]
)
def test_helper_merge_kph(case):
    """the KPH merge (helper h holds chips h kph .. : one histogram over the union, in-place compaction per list,
    per-list tie counts) == the serial merge, and == kph 1 bitwise: the coordinator's round sums, T_g, the per-chip
    quotas, every chip's per-core quotas. GLM53_DSA_SHARD_R8 (8-bit digits: each round's coarse report + the pulled
    bucket) == the 4-bit merge bitwise at every kph -- its per-level sums are the 4-bit rounds' sums, 4 messages fewer;
    the local select (the 32 lists as one lane's 32 chunk cores, k 512 / 37) the same in 4 / 8-bit digits and == the
    sorted top-k"""
    gen = torch.Generator().manual_seed(64 + len(case))
    lists = _lists(gen, case)
    for rank in (0, 5, 7, 18, 31):
        ref = SP.serial_merge(lists, rank)
        base = SP.helper_merge(lists, rank, kph=1)
        for kph in (1, 2, 4):
            got = SP.helper_merge(lists, rank, kph=kph)
            assert (got["T_g"], got["core_quota"]) == (ref["T_g"], ref["core_quota"]), (case, rank, kph)
            for key in ("T_g", "k_rem", "done", "helper_quota", "core_quota", "messages", "hists"):
                assert got[key] == base[key], (case, rank, kph, key)
            r8 = SP.helper_merge(lists, rank, kph=kph, digit_bits=8)
            assert {k: v for k, v in r8.items() if k != "messages"} == {
                k: v for k, v in base.items() if k != "messages"
            }
            assert r8["messages"] == base["messages"] - (0 if base["done"] else 4), (case, rank, kph)
    cores = [[kk for kk, _ in lst] for lst in lists]
    for k in (512, 37):
        sel = {bits: SP.local_select(cores, k, bits) for bits in (4, 8)}
        assert {key: v for key, v in sel[4].items() if key != "messages"} == {
            key: v for key, v in sel[8].items() if key != "messages"
        }, (case, k)
        assert (sel[4]["messages"], sel[8]["messages"]) == (9, 5)
        assert sel[4]["records"] == _topk_records(cores, k) and sel[4]["n_rec"] == min(k, sum(map(len, cores)))
    if case in ("few", "all_empty"):
        assert base["done"] and base["T_g"] == 0
    if case == "ties":
        assert not base["done"] and sum(base["helper_quota"]) == base["k_rem"] > 0  # a real tie round
    if case == "one_chip":
        assert sum(base["helper_quota"][q] for q in range(32) if q != 7) == 0
    if case in ("dups", "adversarial", "boundary"):
        assert not base["done"] and base["k_rem"] > 0 and len(base["hists"]) == 8  # every round + a real tie round


@pytest.mark.parametrize("distinct", [True, False])
def test_lane_merge_kph_lanes(distinct):
    """one token at 1 / 2 / 4 / 8 token lanes with the auto KPH (1 / 1 / 2 / 4): the helpers' merge == the serial
    reference on every chip; with distinct keys the selected pool set is the global top-512 at every lane count"""
    gen = torch.Generator().manual_seed(11)
    n_pools = 32 * 32 * 24  # 24 pool pages per chip: 768 local pools (> 512) -> full local lists
    if distinct:
        keys = torch.randperm(n_pools, generator=gen).to(torch.int64) * 7919 + 13
    else:
        keys = torch.randint(1, 60, (n_pools,), generator=gen, dtype=torch.int64)
    sets = []
    for lanes in (1, 2, 4, 8):
        kph = SP.shard_kph(32, 64, lanes, 8, "auto")
        assert kph == {1: 1, 2: 1, 4: 2, 8: 4}[lanes]
        assert SP.lanes_reasons(min(32, 8 * lanes), lanes, 64, 4, 1, nchips=32, kph=kph) == []
        res = SP.emulate_lane_merge(keys, 32, chunks=64, lanes=lanes, k=512, kph=kph)
        base = SP.emulate_lane_merge(keys, 32, chunks=64, lanes=lanes, k=512, kph=1)
        assert (res["T_g"], res["helper_quota"], res["core_quota"]) == (
            base["T_g"],
            base["helper_quota"],
            base["core_quota"],
        )
        for r in range(32):
            ref = SP.serial_merge(res["lists"], r, k=512)
            assert res["T_g"] == ref["T_g"] and res["core_quota"][r] == ref["core_quota"], (lanes, r)
        r8 = SP.helper_merge(res["lists"], 0, k=512, kph=kph, digit_bits=8)  # GLM53_DSA_SHARD_R8
        assert (r8["T_g"], r8["helper_quota"], r8["core_quota"]) == (
            res["T_g"],
            res["helper_quota"],
            res["core_quota"][0],
        )
        if distinct:  # keys > T_g + the one key == T_g (quota 1): the selected keys = their pools
            sel = {kk for lst in res["lists"] for kk, _ in lst if kk >= res["T_g"]}
            assert len(sel) == 512
            sets.append(sel)
    if distinct:
        top = set(torch.topk(keys, 512).values.tolist())
        assert all(s_ == top for s_ in sets)


def test_kph_lane_rules():
    """8 lanes at t = 16 need KPH 4 (8 helpers in 8 chunk cores); the TL rule (<= 4 tokens per lane -> 4 slots);
    the AG producer count; the knob's values"""
    assert SP.lanes_reasons(16, 8, 64, 4, 1, nchips=32, kph=4) == []
    assert SP.lanes_reasons(16, 8, 64, 4, 1, nchips=32, kph=1) == ["32 merge helpers among 8 lane chunk cores"]
    assert SP.lanes_reasons(16, 4, 64, 4, 1, nchips=32, kph=2) == []
    assert SP.lanes_reasons(16, 8, 64, 4, 1, nchips=32, kph=2) == ["16 merge helpers among 8 lane chunk cores"]
    assert SP.lanes_reasons(32, 8, 64, 4, 1, nchips=32, kph=4) == []  # 4 tokens per lane
    assert SP.lanes_reasons(16, 8, 64, 4, 1, nchips=32, kph=3) == ["kph: a divisor of the 32 chips <= 16"]
    assert SP.shard_tl(32, 8, 4) == 4 and SP.shard_tl(16, 8, 4) == 4 and SP.shard_tl(8, 8, 4) == 4
    assert SP.shard_tl(32, 4, 2) == 8 and SP.shard_tl(16, 2, 1) == 8 and SP.shard_tl(8, 2, 1) == 8  # KPH 1: rev 60's
    assert SP.shard_tl(4, 4, 2) == SP.shard_tl(1, 1, 1) == 4
    assert SP.ag_producers(16, 64, 8) == 16 * (8 + 1) + 1 == 145
    assert SP.mailbox_slots(32, 8) == {lane: [0, 1, 2, 3] for lane in range(8)}
    assert [SP.shard_kph(32, 64, lanes, 8, "auto") for lanes in (1, 2, 4, 8, 16)] == [1, 1, 2, 4, 8]
    assert [SP.shard_kph(32, 64, 8, 8, k) for k in ("1", "2", "4", "16")] == [1, 2, 4, 16]
    assert SP.shard_kph_reserve(32, 8, "auto") == 16 and SP.shard_kph_reserve(32, 8, "4") == 4
    assert SP.shard_kph_reserve(32, 8, "1") == 1
    for bad in ("3", "32", "0"):
        with pytest.raises(ValueError):  # allow-pytest.raises: the unit leg runs --noconftest (no expect_error)
            SP.shard_kph(32, 64, 8, 8, bad)


def _tp_tiles(seq: int, n: int = 32) -> int:
    """a slot's local pooled-key tiles on every chip (tt/dsa_shard.py ShardMap.local_cap -> DsaIndexer.pool_rows)"""
    units = -(-seq // SP.UNIT)
    return -(-units // n) * SP.UNIT // 4 // 32


def test_owner_l1_kph():
    """the owners' CB_AKEY / CB_ASEL / mailbox (tt/shard_plan.py owner_l1: the bytes tt/mk_mla.py allocates) of the
    auto KPH / lanes builds vs today's (KPH 1, 2 lane cores: t = 1 one lane, t >= 2 two), per CB at every t <= 16; t = 32
    has no today build (16 tokens per lane > 8): its total stays under today's t = 16 total"""
    assert SP.HDR["MSTG_PAGES"] == 2 and SP.HDR["KPH"] == 1  # (the header's default)
    assert [SP.merge_scr_bytes(k) for k in (1, 2, 4)] == [14592, 8448, 12544]
    assert SP.merge_scr_bytes(4) + 64 <= SP.merge_scr_bytes(1) + 64  # region U never grows with KPH <= 4
    lanes_new = {1: 1, 2: 2, 4: 4, 8: 8, 16: 8, 32: 8}  # att_lanes with 8 lane cores (LANES_T4 = 4 at t <= 4)
    lanes_old = {1: 1, 2: 2, 4: 2, 8: 2, 16: 2}
    for seq in (32768, 131072, 1048576):
        tp = _tp_tiles(seq)
        old = {t: SP.owner_l1(tp, t, lanes_old[t]) for t in lanes_old}
        top = max(sum(d[k] for k in ("akey", "asel", "mbox")) for d in old.values())
        for t, lanes in lanes_new.items():
            kph = SP.shard_kph(32, 64, lanes, 8, "auto")
            new = SP.owner_l1(tp, t, lanes, kph)
            ref = old[min(t, 16)]
            total = sum(new[k] for k in ("akey", "asel", "mbox"))
            assert total <= top, (seq, t, new, top)
            if t == 32:  # (more CB_ASEL pages, a 4-slot mailbox: 26.2 vs 30.3 KB at 128 K, 44.7 vs 51.8 KB at 1 M)
                assert total <= sum(ref[k] for k in ("akey", "asel", "mbox")), (seq, new, ref)
                continue
            if (seq, t) == (1048576, 4):
                # the one growth: 4 lanes at t = 4 (LANES_T4) halve the lane's chunk cores -> 512 keys per core, the 4
                # kept blocks 2 KB -> 4 KB each (+2 KB CB_AKEY vs today's t = 4; under today's t = 8 / 16 builds)
                assert new["akey"] - ref["akey"] == 2048 and new["asel"] <= ref["asel"] and new["mbox"] == ref["mbox"]
                continue
            for k in ("akey", "asel", "mbox"):
                assert new[k] <= ref[k], (seq, t, k, new, ref)
            if t <= 2:
                assert new == ref  # KPH 1, the same lanes: the same bytes
        for t, lanes in lanes_new.items():  # GLM53_DSA_SHARD_R8: the fine counts in the dropped rounds' mailbox bytes
            kph = SP.shard_kph(32, 64, lanes, 8, "auto")
            off, on = SP.owner_l1(tp, t, lanes, kph), SP.owner_l1(tp, t, lanes, kph, r8=True)
            assert {k: v for k, v in on.items() if k != "mbox"} == {k: v for k, v in off.items() if k != "mbox"}
            assert on["mbox"] == off["mbox"] - (0 if on["tl"] == 4 else 512), (seq, t, on, off)


def test_kph_stays_1_on_the_2x4():
    """the 2x4 (and the pair) keep one chip per helper whatever the knob: their builds are unchanged"""
    assert SP.Geometry(2, 4).P == 2
    for lanes in (1, 2, 4, 8):
        for knob in ("1", "2", "4", "auto"):
            assert SP.shard_kph(8, 64, lanes, SP.Geometry(2, 4).P, knob) == 1
            assert SP.shard_kph(2, 64, lanes, 2, knob) == 1
        t = min(32, 8 * lanes)
        assert SP.lanes_reasons(t, lanes, 64, 4, 1, nchips=8, kph=1) == []
        assert SP.lanes_reasons(t, lanes, 64, 4, 1, nchips=8, kph=2) == ["kph > 1: the 8x4 only"]
        assert SP.shard_tl(t, lanes, 1) == 8  # rev 60's slots
    assert SP.shard_kph_reserve(8, 2, "auto") == 1


# ------------------------------------------------------------------ rev 65: the fold combine + RS1 early (the 8x4)
@contextlib.contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host suite runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


NEG = -1.0e30  # att_c.cpp partial(): an empty partial's m (l = 0, O = 0)


def _slices(gen, n: int, rows: int, *, empty=(), neg_inf=(), d: int = D) -> list:
    """n random partials of `rows` heads (m, l > 0, O); sources in `empty` are the device's empty partial, in
    `neg_inf` the -inf form (torch only: the device never produces it)"""
    out = []
    for s in range(n):
        m = torch.randn(rows, generator=gen) * 4.0
        l_ = torch.rand(rows, generator=gen) * 30.0 + 0.5
        O = torch.randn(rows, d, generator=gen) * l_[:, None]
        if s in empty:
            m, l_, O = torch.full((rows,), NEG), torch.zeros(rows), torch.zeros(rows, d)
        if s in neg_inf:
            m, l_, O = torch.full((rows,), float("-inf")), torch.zeros(rows), torch.zeros(rows, d)
        out.append((m, l_, O))
    return out


@pytest.mark.parametrize("n,rows", [(8, 8), (4, 2), (2, 8)])
@pytest.mark.parametrize("case", ["random", "empty", "neg_inf"])
def test_fold_combine_matches_pairwise(n, rows, case):
    """the fold combine (one joint max, one pass) vs rev 64's pairwise chain (take + n - 1 combines): M bit-equal (max
    is order-free), L / O within fp32 rounding; n = 2 is the same expression -> bit-equal throughout"""
    gen = torch.Generator().manual_seed(65 + n + rows)
    kw = dict(empty=(1, n - 1)) if case == "empty" else dict(neg_inf=(0,)) if case == "neg_inf" else {}
    parts = _slices(gen, n, rows, **kw)
    M, L, O = SP.fold_combine(parts)
    Mc, Lc, Oc = SP.combine_chain(parts)
    assert torch.equal(M, Mc)
    if n == 2:
        assert torch.equal(L, Lc) and torch.equal(O, Oc)
    torch.testing.assert_close(L, Lc, rtol=2e-6, atol=0)
    torch.testing.assert_close(O, Oc, rtol=2e-6, atol=2e-6 * float(Oc.abs().max()))
    assert torch.isfinite(L).all() and torch.isfinite(O).all()
    if case != "random":  # an empty / -inf source contributes exactly nothing
        live = [p for i, p in enumerate(parts) if i not in kw.get("empty", ()) + kw.get("neg_inf", ())]
        M2, L2, O2 = SP.fold_combine(live)
        assert torch.equal(M, M2)
        torch.testing.assert_close(L, L2, rtol=2e-6, atol=0)
    # the kernel's l order (any permutation) keeps L within fp32 rounding
    Lp = SP.fold_combine(parts, l_order=list(reversed(range(n))))[1]
    torch.testing.assert_close(Lp, L, rtol=2e-6, atol=0)


def _rel_err(O, L, o_ref) -> float:
    """mean |O / L - o_ref| / mean |o_ref| (f64)"""
    return float(((O.double() / L.double()[:, None]) - o_ref).abs().mean() / o_ref.abs().mean())


def test_fold_combine_tf32_error_vs_chain():
    """with the FPU's tf32 operands the fold rounds every operand once; the chain re-rounds its accumulator at every
    step: against the f64 reference the fold's error is no larger than the chain's (both well inside bf16's o)"""
    gen = torch.Generator().manual_seed(6565)
    for n, rows in ((8, 8), (4, 2)):
        parts = _slices(gen, n, rows, empty=(3,))
        ref = SP.fold_combine([tuple(v.double() for v in p) for p in parts])
        o_ref = ref[2] / ref[1][:, None]
        M, L, O = SP.fold_combine(parts, tf32_ops=True)
        Mc, Lc, Oc = SP.combine_chain(parts, tf32_ops=True)
        e_fold, e_chain = _rel_err(O, L, o_ref), _rel_err(Oc, Lc, o_ref)
        assert e_fold < 1e-3 and e_chain < 1e-3, (n, e_fold, e_chain)
        assert e_fold <= e_chain * 1.05, (n, e_fold, e_chain)


def _send_tiles(gen, rows_fill=None):
    """a source's send CBs as whole tiles: m, l [32, 32] (column 0 the stats, the other columns finite junk), O
    [16, 32, 32] -- every row finite (the device's send CBs hold whole partials)"""
    m = torch.randn(32, 32, generator=gen) * 4.0
    l_ = torch.rand(32, 32, generator=gen) * 30.0 + 0.5
    O = torch.randn(16, 32, 32, generator=gen) * 50.0
    if rows_fill == "empty":
        m, l_, O = torch.full((32, 32), NEG), torch.zeros(32, 32), torch.zeros(16, 32, 32)
    return m, l_, O


def _junk(gen):
    return lambda: torch.randn(32, 32, generator=gen) * 1e3  # noqa: E731  (the receive CB's stale, finite content)


def _rows_of(tiles_O, r0: int, rows: int):
    """rows r0 .. r0 + rows of the 16 O tiles -> [rows, 512]"""
    return torch.cat([tiles_O[d][r0 : r0 + rows] for d in range(16)], dim=1)


@pytest.mark.parametrize("j1", [0, 1, 2, 3])
def test_fold1_tiles_layout(j1):
    """att_dsa_shard_dm1.cpp land_fold1 + att_c.cpp fold1 (rev 66), emulated on whole face-major tiles (the m rotations,
    the packed l, the row-packed O pages, FOLD1): rows 8 j1 .. of the send CBs = the fold combine of the 8 sources'
    slices (M bit-equal; L / O within fp32 rounding: L is FOLD1's matmul sum now); L's / O's other rows exactly 0 (the RS2
    ship never reads them)"""
    gen = torch.Generator().manual_seed(100 + j1)
    src = [_send_tiles(gen, "empty" if s == 5 else None) for s in range(8)]
    M, L, O = SP.fold1_tiles(src, j1, _junk(gen))
    rows = slice(8 * j1, 8 * j1 + 8)
    parts = [(m[rows, 0], l_[rows, 0], _rows_of(o, 8 * j1, 8)) for m, l_, o in src]
    Mf, Lf, Of = SP.fold_combine(parts, l_order=SP.fold1_l_order(j1))
    assert torch.equal(M[rows, 0], Mf)
    torch.testing.assert_close(L[rows, 0], Lf, rtol=2e-6, atol=0)
    torch.testing.assert_close(_rows_of(O, 8 * j1, 8), Of, rtol=1e-5, atol=1e-5 * float(Of.abs().max()))
    other = torch.ones(32, dtype=torch.bool)
    other[rows] = False
    assert torch.equal(O[:, other], torch.zeros_like(O[:, other])) and torch.equal(L[other], torch.zeros_like(L[other]))
    assert torch.isfinite(M).all() and torch.isfinite(L).all()  # every column: both runs of the stats land
    # the pairwise chain of rev 64 (own slice first, then the column peers in position order) agrees within fp32
    for own in (j1, 4 + j1):
        order = [own] + [s for s in range(8) if s != own]
        Mc, Lc, Oc = SP.combine_chain([parts[s] for s in order])
        assert torch.equal(Mc, Mf)
        torch.testing.assert_close(Lf, Lc, rtol=2e-6, atol=0)
        torch.testing.assert_close(Of, Oc, rtol=1e-5, atol=1e-5 * float(Oc.abs().max()))


@pytest.mark.parametrize("j1,lpos", [(0, 0), (1, 3), (2, 1), (3, 2)])
@pytest.mark.parametrize("stale", ["random", "huge", "empty_stats"])
def test_fold2_tiles_layout(j1, lpos, stale):
    """att_dsa_shard_dm1.cpp land_fold2 + att_c.cpp fold2 (round 2 + the finalise): rows 2 j2 .. of the bf16-bound o
    tiles = the fold of the 4 row sources' 2-row slices / L, against rev 64's chain + finalise within fp32 rounding --
    whatever the stale rows of the landed pages hold (finite: large O values, the -1e30 of an empty partial's m)"""
    gen = torch.Generator().manual_seed(200 + 4 * j1 + lpos)
    src = [_send_tiles(gen) for _ in range(4)]
    junk = {
        "random": _junk(gen),
        "huge": lambda: torch.randn(32, 32, generator=gen) * 3e4,  # noqa: E731
        "empty_stats": lambda: torch.where(torch.rand(32, 32, generator=gen) < 0.5, NEG, 0.0),  # noqa: E731
    }[stale]
    o = SP.fold2_tiles(src, j1, lpos, junk)
    j2 = 4 * j1 + lpos
    rows = slice(2 * j2, 2 * j2 + 2)
    parts = [(m[rows, 0], l_[rows, 0], _rows_of(O, 2 * j2, 2)) for m, l_, O in src]
    Mf, Lf, Of = SP.fold_combine(parts, l_order=SP.fold2_l_order(lpos))
    want = Of / Lf[:, None]
    got = _rows_of(o, 2 * j2, 2)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5 * float(want.abs().max()))
    order = [lpos] + [s for s in range(4) if s != lpos]
    Mc, Lc, Oc = SP.combine_chain([parts[s] for s in order])
    torch.testing.assert_close(got, Oc / Lc[:, None], rtol=1e-5, atol=1e-5 * float(want.abs().max()))


def test_fold_tiles_tf32_close():
    """the whole-tile emulation with tf32 matmul operands (FOLD / diag coefficients exact in tf32, O and c l rounded
    once) stays within tf32 rounding of the fp32 fold; M never passes the FPU"""
    gen = torch.Generator().manual_seed(303)
    src = [_send_tiles(gen) for _ in range(8)]
    M, L, O = SP.fold1_tiles(src, 2, _junk(gen))
    gen = torch.Generator().manual_seed(303)
    src = [_send_tiles(gen) for _ in range(8)]
    Mt, Lt, Ot = SP.fold1_tiles(src, 2, _junk(gen), tf32_ops=True)
    assert torch.equal(M, Mt)
    torch.testing.assert_close(Lt, L, rtol=1e-3, atol=0)
    torch.testing.assert_close(Ot, O, rtol=3e-3, atol=3e-3 * float(O.abs().max()))


def _real_send(gen, src_count: int, empty=()) -> list:
    """send tiles with a real cache's stats: m ~ the scaled logits' row max (around 20, spread 5 between chips), l in
    [1, 500), O = l-weighted values; `empty` sources are the device's empty partial"""
    out = []
    for s in range(src_count):
        m = 20.0 + 5.0 * torch.randn(32, 1, generator=gen) + 0.01 * torch.randn(32, 32, generator=gen)
        l_ = 1.0 + 499.0 * torch.rand(32, 1, generator=gen) + torch.zeros(32, 32)
        O = torch.randn(16, 32, 32, generator=gen) * l_[None, :, :1]
        if s in empty:
            m, l_, O = torch.full((32, 32), NEG), torch.zeros(32, 32), torch.zeros(16, 32, 32)
        out.append((m, l_, O))
    return out


def test_fold_real_stats_need_exact_fp32():
    """the rev-65 failure class on the host: with a real cache's stats (m ~ 20) a stats path through Tf32-dest CBs
    (rev 65: the m / l / c tiles loaded from CB_ARXO / CB_APAO) weighs each source by exp(m - tf32(m)) -- percent-level
    errors in the column-combined slice; rev 66's exact stats (CB_ARX / CB_APB / CB_ACOEF) with the FPU's tf32 matmul
    operands stay at the tf32 floor (and below rev 64's chain). A zero cache (m = 0 everywhere) hides both."""
    gen = torch.Generator().manual_seed(6666)
    src = _real_send(gen, 8, empty=(6,))
    j1 = 1
    rows = slice(8 * j1, 8 * j1 + 8)
    parts = [(m[rows, 0], l_[rows, 0], _rows_of(o, 8 * j1, 8)) for m, l_, o in src]
    ref = SP.fold_combine([tuple(v.double() for v in p) for p in parts])
    o_ref = ref[2] / ref[1][:, None]

    def err(stats_tf32):
        g2 = torch.Generator().manual_seed(1)
        M, L, O = SP.fold1_tiles(src, j1, _junk(g2), tf32_ops=True, stats_tf32=stats_tf32)
        return _rel_err(_rows_of(O, 8 * j1, 8), L[rows, 0], o_ref)

    e_v66, e_v65 = err(False), err(True)
    Mc, Lc, Oc = SP.combine_chain(parts, tf32_ops=True)
    e_chain = _rel_err(Oc, Lc, o_ref)
    assert e_v66 < 1e-3 and e_v66 <= e_chain * 1.05, (e_v66, e_chain)
    assert e_v65 > 4 * e_v66, (e_v65, e_v66)  # the exp(m - M) amplification of tf32 stats


_REV65_FOLD_EXCERPT = """
#if defined(MLA_DSA_SHARD_COMB)
FORCE_INLINE void max_pages(uint32_t n) {
    load(CB_ARXO, 0, 0);
}
FORCE_INLINE void fold1(Part out) {
    matmul_init(CB_ARX, CB_ADIAG, 0);
    matmul_tiles(CB_ARX, CB_ADIAG, 0, 0, 0);
    matmul_tiles(CB_APB, CB_ARXO, 0, 2 * j, j);
    load(CB_APAO, q, 1);
}
#endif
"""


def test_att_c_format_rule():
    """THE FORMAT RULE (rev 66): att_c.cpp takes every matmul operand from a Tf32-dest CB and loads the fold combine's
    stats from Float32-dest CBs only -- rev 65's fold (excerpt) breaks both (its device run decoded token 0 every step)
    """
    assert SP.att_c_format_audit() == []
    bad = SP.att_c_format_audit(_REV65_FOLD_EXCERPT)
    assert [n for n, _ in bad] == [4, 7, 8, 9, 10], bad
    assert (
        "CB_ARXO" in bad[0][1] and "operand CB_ARX:" in bad[1][1] and "CB_APB" in bad[3][1] and "CB_APAO" in bad[4][1]
    )
    assert SP.att_c_format_audit("matmul_tiles(CB_ADIAG, a.s, 0, 0, 0);") != []  # a Part's stats CB


def test_exchange_fold_matches_direct_and_pairwise():
    """the whole 8x4 exchange with the fold combine in both rounds: = the direct 32-chip attention (the tolerance of
    test_exchange_matches_direct) and = rev 64's pairwise exchange within fp32 rounding"""
    gen = torch.Generator().manual_seed(1234 + 65)
    g = SP.Geometry(8, 4)
    parts, exact = _direct_parts(gen, g.n)
    q = torch.randn(64, D, generator=gen)
    fold = SP.emulate_exchange(g, q, parts, comb="fold")
    pair = SP.emulate_exchange(g, q, parts)
    ref = exact(q)
    for r in range(g.n):
        torch.testing.assert_close(fold[r], ref[g.hpc * r : g.hpc * (r + 1)], atol=2e-4, rtol=1e-4)
        torch.testing.assert_close(fold[r], pair[r], atol=1e-5, rtol=1e-5)
    with expect_error(AssertionError, "the fold combine: the 8x4"):
        SP.emulate_exchange(SP.Geometry(2, 4), q, parts, comb="fold")


@pytest.mark.parametrize("t", [1, 4, 16])
def test_rs1_early_plan(t):
    """RS1 early: every 8x4 chip's column scatter sends the other half first (pass 0's destinations: the tile pass 0
    attends with), then its own half -- the same destinations, hops and directions as rev 64's order"""
    g = SP.Geometry(8, 4)
    for r in range(8):
        for c in range(4):
            rank = g.rank(r, c)
            base = g.block("rs1", t, r, c)
            early = g.block("rs1", t, r, c, rs1_early=True)
            assert sorted(early["dsts"]) == sorted(base["dsts"])
            first = SP.rs1_early_first(r, 8)
            assert [d[0] for d in early["dsts"][: len(first)]] == first and len(first) == 4
            assert all(q // 4 == g.pass_tile(rank, 0) for q in first)  # = pass 0's slices
            assert all(q // 4 == g.tile(rank) for q, _, _, _ in early["dsts"][4:])  # = pass 1's (own half, not self)
            assert {k: v for k, v in early.items() if k != "dsts"} == {k: v for k, v in base.items() if k != "dsts"}
            assert g.block("rs2", t, r, c, rs1_early=True) == g.block("rs2", t, r, c)
    with expect_error(AssertionError, "RS1 early: the 8x4 column scatter"):
        SP.Geometry(2, 4).block("rs1", 4, 0, 0, rs1_early=True)


def test_rs1_early_gates_two_fields():
    """rev 67, the RS1-early race (B 16 distinct users, 8 lanes x 2 tokens: user 0 moved at decode id 8): with one count
    of every root signal (rev 65 / 66) the first gate opens at t signals in total -- the fast lanes' pass-1 signals open
    it while a slow lane's pass-0 slices are unwritten; with two fields (rev 67) it opens exactly at the last pass-0
    signal. Balanced lanes (B 4: 4 identical users x 1 token) never expose it, which is why B 4 stayed bit-identical."""
    # 8 lanes x 2 tokens, lane 7 (more selected pools) slow in pass 0
    pass_us = [([40.0, 40.0], [50.0, 50.0])] * 7 + [([200.0, 200.0], [50.0, 50.0])]
    done0, done1 = SP.rs1_lane_ships(pass_us)
    need1, need2 = max(max(d) for d in done0), max(max(d) for d in done1)
    g1_old, g2_old = SP.rs1_early_gates(done0, done1, 16, fields=False)
    g1, g2 = SP.rs1_early_gates(done0, done1, 16)
    assert g1_old == 130.0 < need1 == 400.0  # lanes 0..6 through both passes: 7 x 2 + 7 > 16 signals
    assert (g1, g2) == (need1, need2)
    # B 4: 4 lanes x 1 token, identical users -> the one-count gate happens to be sound
    done0, done1 = SP.rs1_lane_ships([([60.0], [55.0])] * 4)
    assert SP.rs1_early_gates(done0, done1, 4, fields=False) == SP.rs1_early_gates(done0, done1, 4) == (60.0, 115.0)
    # ... but one token per lane races too once the lanes skew (2 lanes through both passes = t = 4 signals)
    done0, done1 = SP.rs1_lane_ships([([30.0], [30.0])] * 2 + [([100.0], [30.0])] * 2)
    assert (
        SP.rs1_early_gates(done0, done1, 4, fields=False)[0] == 60.0 < 100.0 == SP.rs1_early_gates(done0, done1, 4)[0]
    )


def test_rs1_early_gates_sound_random():
    """rev 67's two-field gates are sound for any lane skew (gate 1 at the last pass-0 signal, gate 2 at the last
    pass-1 signal); the one-count gates of rev 65 / 66 are not"""
    gen = torch.Generator().manual_seed(67)
    unsound_old = 0
    for _ in range(200):
        lanes = int(torch.randint(1, 9, (1,), generator=gen))
        per = int(torch.randint(1, 5, (1,), generator=gen))
        pass_us = [
            ((torch.rand(per, generator=gen) * 100 + 1).tolist(), (torch.rand(per, generator=gen) * 100 + 1).tolist())
            for _ in range(lanes)
        ]
        done0, done1 = SP.rs1_lane_ships(pass_us)
        t = lanes * per
        need1, need2 = max(max(d) for d in done0), max(max(d) for d in done1)
        assert SP.rs1_early_gates(done0, done1, t) == (need1, need2)
        g1_old, _ = SP.rs1_early_gates(done0, done1, t, fields=False)
        unsound_old += g1_old < need1
    assert unsound_old > 20, unsound_old


def test_rev65_knobs(monkeypatch):
    """GLM53_DSA_SHARD_COMB / GLM53_DSA_SHARD_RS1_EARLY / GLM53_DSA_SHARD_RSB / GLM53_DSA_SHARD_R8 /
    GLM53_DSA_SHARD_RSNP: default off, the 8x4 only, 0 / 1 only (RSB / R8 need no other knob, RSNP needs RSB); the fold's
    message size is the header's; R8's rounds, mailbox layout (the fine counts after the 4-round messages: 16-B aligned,
    never more bytes than the 4-bit mailbox) and the merge message count are the kernels'; RSNP's data pages per slice
    (9 of 10 / 3 of 4) are the bytes the roots write and land (checked on the kernels' text)"""
    for k in ("GLM53_DSA_SHARD_COMB", "GLM53_DSA_SHARD_RS1_EARLY", "GLM53_DSA_SHARD_RSB", "GLM53_DSA_SHARD_R8",
              "GLM53_DSA_SHARD_RSNP"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    assert not SP.shard_comb(8) and not SP.shard_rs1_early(8) and not SP.shard_rsb(8) and not SP.shard_r8(8)
    assert not SP.shard_rsnp(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_R8", "1")
    assert SP.shard_r8(8) and not SP.shard_r8(2)  # on its own (the select / merge kernels and the mailbox only)
    assert not (SP.shard_comb(8) or SP.shard_rs1_early(8) or SP.shard_rsb(8) or SP.shard_agk(8))
    SP.r8_header_check()
    assert (SP.radix_rounds(), SP.radix_rounds(True), SP.merge_msgs(32), SP.merge_msgs(32, r8=True)) == (8, 4, 17, 13)
    assert [(SP.mbox_bytes(tl, 32), SP.mbox_bytes(tl, 32, r8=True), SP.r8_fine_off(tl, 32)) for tl in (4, 8)] == [
        (4800, 4800, 4288),
        (9600, 9088, 8576),
    ]
    monkeypatch.setenv("GLM53_DSA_SHARD_R8", "yes")
    with expect_error(ValueError, "GLM53_DSA_SHARD_R8=yes: 0 or 1"):
        SP.shard_r8(8)
    monkeypatch.delenv("GLM53_DSA_SHARD_R8")
    monkeypatch.setenv("GLM53_DSA_SHARD_RSNP", "1")
    with expect_error(ValueError, "GLM53_DSA_SHARD_RSNP=1 needs GLM53_DSA_SHARD_RSB=1"):
        SP.shard_rsnp(8)
    assert not SP.shard_rsnp(2)  # the pair / the 2x4: never (no RSB there either)
    monkeypatch.setenv("GLM53_DSA_SHARD_RSB", "1")
    assert SP.shard_rsb(8) and not SP.shard_rsb(2)  # on its own (the scatters only)
    assert SP.shard_rsnp(8) and not SP.shard_rsnp(2)
    monkeypatch.setenv("GLM53_DSA_SHARD_RSNP", "yes")
    with expect_error(ValueError, "GLM53_DSA_SHARD_RSNP=yes: 0 or 1"):
        SP.shard_rsnp(8)
    monkeypatch.delenv("GLM53_DSA_SHARD_RSNP")
    SP.rsnp_header_check()
    assert (SP.RS1_DATA_PAGES, SP.RS2B_DATA_PAGES) == (9, 3)  # + one pad page each: RS2_PPC = 10, RS2B_PPC = 4
    assert [SP.rs_slice_bytes(r, otb) for r in (8, 2) for otb in (4096, 2048)] == [18432, 10240, 4608, 2560]
    with expect_error(AssertionError, "RSNP: on RSB's block chunks"):
        SP.rs_stream_args(4, rsnp=True)
    monkeypatch.setenv("GLM53_DSA_SHARD_COMB", "1")
    monkeypatch.setenv("GLM53_DSA_SHARD_RS1_EARLY", "1")
    assert SP.shard_comb(8) and SP.shard_rs1_early(8)
    assert not SP.shard_comb(2) and not SP.shard_rs1_early(2)  # the pair / the 2x4 keep rev 64's kernels
    monkeypatch.setenv("GLM53_DSA_SHARD_COMB", "2")
    with expect_error(ValueError, "GLM53_DSA_SHARD_COMB=2: 0 or 1"):
        SP.shard_comb(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_RSB", "on")
    with expect_error(ValueError, "GLM53_DSA_SHARD_RSB=on: 0 or 1"):
        SP.shard_rsb(8)
    assert SP.HDR["COMB_MSG"] == 8


# ------------------------------------------------------------------ rev 68: the key-packed all-gather (AGK)
def _stale(rnd) -> list:
    """an earlier launch's leftovers in an x_ag chunk (nonzero words everywhere)"""
    return [rnd.randrange(1, 1 << 32) for _ in range(SP.AG32_PPC * SP.WORDS)]


@pytest.mark.parametrize("case", ["random", "ties", "few", "empty", "one_chip", "all_empty"])
def test_keypack_merge_matches_records(case):
    """fs_read.cpp FS_KEYPACK's in-place pack of each chip's AG32 chunk (stale words past its records) -> its key page
    (the keys in record order, zeros after; the q page verbatim); the AGK merge (every list from its key page with its
    count, this chip's tie order from its own x_ag records) == rev 65's helper_merge over the record lists bitwise (round
    sums, T_g, k_rem, done, per-chip / per-core quotas, messages) and == the serial merge, at kph 1 / 2 / 4"""
    gen = torch.Generator().manual_seed(660 + len(case))
    rnd = random.Random(len(case))
    lists = _lists(gen, case)
    chunks = [SP.x_ag_chunk(lst, q_words=[rnd.randrange(1 << 32) for _ in range(SP.WORDS)], stale=_stale(rnd))
              for lst in lists]  # fmt: skip
    packed = [SP.keypack_chunk(ch) for ch in chunks]
    kp = slice(SP.AG32_QPAGES * SP.WORDS, SP.AGK_PPC * SP.WORDS)
    for lst, pk, ch in zip(lists, packed, chunks):
        page = pk[kp]
        assert page[: len(lst)] == [kk for kk, _ in lst] and not any(page[len(lst) :])
        assert pk[: SP.WORDS] == ch[: SP.WORDS]  # the q page travels verbatim
        assert SP.agk_records(ch) == [tuple(r) for r in lst]
    pages, counts = [pk[kp] for pk in packed], [len(lst) for lst in lists]
    for rank in (0, 5, 7, 18, 31):
        ref = SP.serial_merge(lists, rank)
        for kph in (1, 2, 4):
            base = SP.helper_merge(lists, rank, kph=kph)
            got = SP.agk_merge(pages, counts, chunks[rank], rank, kph=kph)
            assert got == base, (case, rank, kph)
            r8 = SP.agk_merge(pages, counts, chunks[rank], rank, kph=kph, digit_bits=8)  # GLM53_DSA_SHARD_R8
            assert {k: v for k, v in r8.items() if k != "messages"} == {
                k: v for k, v in base.items() if k != "messages"
            }
            assert (got["T_g"], got["core_quota"]) == (ref["T_g"], ref["core_quota"]), (case, rank, kph)
    # a wrong count is caught by the helpers' checks (the kernel's watcher ASSERTs)
    if case == "random":
        with expect_error(AssertionError, "the count runs short of the keys"):
            SP.agk_keys(pages[0], counts[0] - 1)
        with expect_error(AssertionError, "a zero key inside the list"):
            SP.agk_keys(packed[1][kp][:-1] + [0], SP.CAND_MAX)


def _launch(gen, case: str, T: int) -> tuple:
    """positions + order keys of T sparse tokens: "sparse" (16 .. 32 local pools per chip), "random" / "ties" (mixed
    lengths up to 192 per chip, at T <= 2 token 0 with full 512-record lists on every chip; ties: 40 key values)"""
    pos, keys = [], []
    for t in range(T):
        if case == "sparse":
            p = int(torch.randint(2051, 4096, (1,), generator=gen))
        elif t == 0 and T <= 2:
            p = 65535 + 4096 * int(torch.randint(0, 3, (1,), generator=gen))  # >= 512 local pools on every chip
        else:
            p = int(torch.randint(2051, 24576 if T <= 8 else 8192, (1,), generator=gen))
        hi = 40 if case == "ties" else 1 << 31
        keys.append(torch.randint(1, hi, ((p + 1) // 4,), generator=gen, dtype=torch.int64))
        pos.append(p)
    return pos, keys


@pytest.mark.parametrize("case", ["random", "ties", "sparse"])
@pytest.mark.parametrize("tl", [1, 2, 4])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_agk_launch_matches_rev65(lanes, tl, case):
    """one launch of t = lanes x TL tokens under AGK (auto KPH 1 / 1 / 2 / 4): every chip's local top-512 records ->
    x_ag (stale words past them) -> the key pack + both AG rounds -> y_ag (32 blocks x t x 2 pages, each written once)
    -> ix_dm1's q pieces where x_ag had them, and on chips 0 / 13 / 31 the AGK merge == rev 65's helper merge over the
    records bitwise (T_g, per-chip / per-core quotas, the tie order) == the serial merge, per token, in 4-bit and in
    R8's 8-bit digits; the count derived from the position = the local select's record count on every chip"""
    t = lanes * tl
    kph = SP.shard_kph(32, 64, lanes, 8, "auto")
    gen = torch.Generator().manual_seed(1000 * lanes + 10 * tl + len(case))
    pos, keys = _launch(gen, case, t)
    ranks = (0, 13, 31) if t <= 8 else (0, 31)
    res = SP.emulate_agk_launch(keys, pos, lanes, kph, ranks=ranks, seed=t, r8=True)
    assert [r["lane"] for r in res] == [i % lanes for i in range(t)]
    for i, r in enumerate(res):
        for rank in ranks:
            assert r["got"][rank] == r["ref"][rank], (lanes, tl, case, i, rank)
            g8, g4 = r["r8"][rank], r["got"][rank]  # GLM53_DSA_SHARD_R8: the same merge in 8-bit digits
            assert {k: v for k, v in g8.items() if k != "messages"} == {k: v for k, v in g4.items() if k != "messages"}
            assert g8["messages"] == g4["messages"] - (0 if g4["done"] else 4)
            ser = r["serial"][rank]
            assert (r["got"][rank]["T_g"], r["got"][rank]["core_quota"]) == (ser["T_g"], ser["core_quota"])
        assert sum(r["got"][0]["helper_quota"]) == r["got"][0]["k_rem"]  # the tie quotas cover k_rem exactly
        if case == "ties":
            assert r["got"][0]["k_rem"] > 0 and not r["got"][0]["done"]  # a real tie round on every token


def test_agk_count_is_the_local_select_count():
    """the AGK count (min(512, n_closed_local(p, q))) = the local top-512's record count on every chip at every lane
    count, across pool-page / unit boundaries, below / at / above 512 local pools (the key page carries no count)"""
    gen = torch.Generator().manual_seed(66)
    for p in (2051, 2052, 4095, 4096 + 127, 4096 + 128 + 3, 32767, 65535 - 128, 65535, 65535 + 129, 262143):
        keys = torch.randint(1, 1 << 31, ((p + 1) // 4,), generator=gen, dtype=torch.int64)
        g = torch.arange(int(keys.numel()))
        for q in (0, 1, 15, 31):
            mine = torch.nonzero((g // SP.POOL_PAGE) % 32 == q).reshape(-1)
            assert int(mine.numel()) == SP.n_closed_local(p, q, 32), (p, q)
            lq = (mine // SP.POOL_PAGE // 32) * SP.POOL_PAGE + mine % SP.POOL_PAGE
            for lanes in (1, 8):
                assert len(SP.lane_records(keys[mine], lq, 64, lanes)) == SP.agk_count(p, q), (p, q, lanes)


def test_rev68_agk_knob_and_layout(monkeypatch):
    """GLM53_DSA_SHARD_AGK: default off, the 8x4 only, 0 / 1 only; the kernels' AGK layout / fs_read.cpp's record format
    = the host mirrors; the busiest link-0 hop of one launch carries 7 + 3 x 8 blocks of t chunks: 6 pages each -> 2"""
    monkeypatch.delenv("GLM53_DSA_SHARD_AGK", raising=False)
    assert not SP.shard_agk(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_AGK", "1")
    assert SP.shard_agk(8) and not SP.shard_agk(2)  # the pair / the 2x4 keep their AG formats
    monkeypatch.setenv("GLM53_DSA_SHARD_AGK", "2")
    with expect_error(ValueError, "GLM53_DSA_SHARD_AGK=2: 0 or 1"):
        SP.shard_agk(8)
    SP.agk_header_check()
    assert SP.AGK_PPC == 2 and SP.FSR["KP_CAND_MAX"] * 4 == SP.TB
    geo, T = SP.Geometry(8, 4), 16
    assert SP.agy_q_page(T, 3, 5, 15) == ((3 * T + 5) * 2, 15 * 128) and SP.agk_key_page(T, 3, 5) == (3 * T + 5) * 2 + 1
    hop = (geo.P - 1) + (geo.L - 1) * geo.P  # AG1's last column hop + AG2's last row hop (multicast lines)
    assert hop == 31
    assert hop * T * SP.AG32_PPC * SP.TB == 3 * hop * T * SP.AGK_PPC * SP.TB == 6094848  # 5.8 MB -> 1.9 MB at t 16


@pytest.mark.parametrize("t", [1, 2, 3, 4, 8, 16, 32])
def test_block_chunks_keep_pages(t):
    """GLM53_DSA_SHARD_AGB: AG1's packed pages land on AGK's page ids (agy_q_page / agk_key_page: no consumer change),
    every y_ag page once, one signal per block (its last token's pair); AG2's 8-page batches of the block chunk cover
    the same pages with one signal per block; the batch / signal counts per launch. GLM53_DSA_SHARD_RSB: the 8x4
    scatters' block chunks send every chip's page pairs in the knob-off order (RS1 early's order too) from the roots'
    rs1_page / rs2b_page chunks to the landing roots' reads of source block p, each landed page once, the chunk-end
    signal on each destination's last pair only; the staging CBs keep their bytes. GLM53_DSA_SHARD_RSNP (on RSB): the
    same lanes stage / send only the slices' data pages (fs_scatter_packets: fs_read.cpp + fs_send.cpp FS_SUBPC /
    FS_SUBD) -- every page goes from RSB's x page to RSB's y page, in RSB's order, all data pages and no pad page (odd t:
    the last slice's first pad page closes the last pair, as in RSB), the signal on each destination's last pair; the
    landed y equals RSB's (x's pads zero) and the bytes the landing roots read are identical; 9 / 10 and 3 / 4 of the
    bytes and packets at even t, the staging CB within RSB's bytes"""
    T, seen = t, {}
    for blk in range(32):
        sends = SP.agb_ag1_sends(T, blk)
        assert [pg for pg, _ in sends] == [SP.agk_key_page(T, blk, c) - 1 for c in range(T)]  # the pair starts
        assert all(SP.agy_q_page(T, blk, c, d)[0] == sends[c][0] for c in range(T) for d in (0, 15))
        assert [sig for _, sig in sends] == [False] * (T - 1) + [True]
        for pg, _ in sends:
            for k in range(SP.AGK_PPC):
                assert pg + k not in seen
                seen[pg + k] = blk
        yp = SP.AGK_PPC * T
        nb = 8 if yp % 8 == 0 else (4 if yp % 4 == 0 else 2)
        ag2 = SP.agb_ag2_sends(T, blk, nb)
        assert [pg for pg, _ in ag2] == [pg for pg, _ in sends] and [s_ for _, s_ in ag2] == [s_ for _, s_ in sends]
    assert sorted(seen) == list(range(32 * T * SP.AGK_PPC))
    base, agb = SP.ag_stream_counts(t, agk=True), SP.ag_stream_counts(t, agk=True, agb=True, nb2=nb)
    assert (base["ag1_signals"], base["ag2_signals"], agb["ag1_signals"], agb["ag2_signals"]) == (t, 8 * t, 1, 8)
    assert agb["ag2_batches"] == 8 * SP.AGK_PPC * t // nb
    if t == 16:
        assert (base["ag1_batches"] + base["ag2_batches"], agb["ag1_batches"] + agb["ag2_batches"]) == (144, 48)
    g = SP.Geometry(8, 4)
    for name, axis, ppc, page in (("rs1", g.pair_axis, SP.RS2_PPC, SP.rs1_page),
                                  ("rs2", g.line_axis, SP.RS2B_PPC, SP.rs2b_page)):  # fmt: skip
        n, arg = g.axis_len(axis), {rsb: SP.rs_stream_args(t, rsb=rsb)[name] for rsb in (False, True)}
        for p in range(n):
            for first in (None, SP.rs1_early_first(p, 8)) if name == "rs1" else (None,):
                sends = {}
                for rsb, a in arg.items():
                    *_, dsts = SP.fs_scatter_block(p, n, a["rows"] // 32, first)
                    sends[rsb] = SP.fs_scatter_sends(a["rows"], a["width"], a["extra"], p, dsts)
                order = [q for q, _, _, _ in dsts]
                want = [(q, page(t, q, tok) + c, page(t, p, tok) + c) for q in order for tok in range(t)
                        for c in range(0, ppc, 2)]  # fmt: skip
                assert [s_[:3] for s_ in sends[False]] == [s_[:3] for s_ in sends[True]] == want
                assert len({(q, y) for q, _, y in want}) == len(want)  # every landed pair once
                # the chunk-end signal: knob off on every token's last pair, RSB on each destination's last pair
                assert [s_[3] for s_ in sends[False]] == [x % ppc == ppc - 2 for _, x, _ in want]
                assert [s_[3] for s_ in sends[True]] == [x == page(t, q, t - 1) + ppc - 2 for q, x, _ in want]
        off, on = SP.rs_stream_counts(t)[name], SP.rs_stream_counts(t, rsb=True)[name]
        assert (off["signals"], on["signals"]) == ((n - 1) * t, n - 1)  # chunk-end signals per source
        assert (off["wait_per_source"], on["wait_per_source"]) == (t, 1)  # the waiter's count per source lane
        assert on["packets"] == off["packets"] == (n - 1) * t * ppc // 2 and on["cb_bytes"] == off["cb_bytes"]
        assert on["batches"] == (off["batches"] // 2 if (name == "rs2" and t % 2 == 0) else off["batches"])
        # RSNP: the RSB lanes without the pad pages
        data = SP.RS1_DATA_PAGES if name == "rs1" else SP.RS2B_DATA_PAGES
        a_b, a_n = SP.rs_stream_args(t, rsb=True)[name], SP.rs_stream_args(t, rsb=True, rsnp=True)[name]
        assert (a_n["rows"], a_n["width"]) == (a_b["rows"], a_b["width"]) and a_n["extra"]["sub"] == (ppc, data)
        st_b, st_n = (SP.fs_stage(a["rows"], a["width"], a["extra"]) for a in (a_b, a_n))
        assert (
            st_n["spc"] == SP.rs_sent_pages(t, data) == t * data + (t * data) % 2
            and st_n["cb_bytes"] <= st_b["cb_bytes"]
        )
        for p in range(n):
            for first in (None, SP.rs1_early_first(p, 8)) if name == "rs1" else (None,):
                *_, dsts = SP.fs_scatter_block(p, n, 1, first)
                pk_b = SP.fs_scatter_packets(a_b["rows"], a_b["width"], a_b["extra"], p, dsts)
                pk_n = SP.fs_scatter_packets(a_n["rows"], a_n["width"], a_n["extra"], p, dsts)
                sends_b = SP.fs_scatter_sends(a_b["rows"], a_b["width"], a_b["extra"], p, dsts)
                assert [(q, xs[0], ys[0], sig) for q, xs, ys, sig in pk_b] == sends_b  # the two RSB ports agree
                assert all(xs[1] == xs[0] + 1 and ys[1] == ys[0] + 1 for _, xs, ys, _ in pk_b)
                pg_b = [(q, xs[j], ys[j]) for q, xs, ys, _ in pk_b for j in (0, 1)]
                pg_n = [(q, xs[j], ys[j]) for q, xs, ys, _ in pk_n for j in (0, 1)]
                fill = page(t, p, t - 1) + data if (t * data) % 2 else None  # odd t: the pair filler
                assert pg_n == [e for e in pg_b if e[2] % ppc < data or e[2] == fill]
                assert all(x_ - page(t, q, 0) == y_ - page(t, p, 0) for q, x_, y_ in pg_n)  # x page -> its RSB y page
                lastq = {q: i for i, (q, _, _, _) in enumerate(pk_n)}
                assert [sig for *_, sig in pk_n] == [i == lastq[q] for i, (q, _, _, _) in enumerate(pk_n)]
                # landing: random x data pages, zero pads (ttnn.zeros, never written by the roots); y zeros
                gen = torch.Generator().manual_seed(1000 * p + t)
                xv = torch.randint(1, 1 << 30, (n * t * ppc,), generator=gen)
                xv[torch.arange(n * t * ppc) % ppc >= data] = 0
                y_b, y_n = {}, {}
                for pg_, yd in ((pg_b, y_b), (pg_n, y_n)):
                    for q, x_, y_ in pg_:
                        yd[(q, y_)] = int(xv[x_])
                assert {k: v for k, v in y_b.items() if v} == {k: v for k, v in y_n.items() if v}  # y bit-identical
                for q, _, _, _ in dsts:  # the landing root of q reads [0, DT ob + ST pb) of block p's slices
                    for otb in (4096, 2048):
                        nbytes = SP.rs_slice_bytes(8 if name == "rs1" else 2, otb)
                        for tok in range(t):
                            for b in range(0, nbytes, 256):
                                y_ = page(t, p, tok) + b // SP.TB
                                assert y_n[(q, y_)] == y_b[(q, y_)] == int(xv[page(t, q, tok) + b // SP.TB])
        c_n = SP.rs_stream_counts(t, rsb=True, rsnp=True)[name]
        assert (c_n["signals"], c_n["wait_per_source"]) == (on["signals"], on["wait_per_source"]) == (n - 1, 1)
        assert c_n["packets"] * 2 == c_n["bytes"] // SP.TB == (n - 1) * SP.rs_sent_pages(t, data)
        if t % 2 == 0:  # 9 / 10 (RS1), 3 / 4 (RS2) of RSB's bytes, packets and middle-link bytes
            assert c_n["bytes"] * ppc == on["bytes"] * data and c_n["packets"] * ppc == on["packets"] * data
            assert c_n["mid_bytes"] * ppc == on["mid_bytes"] * data
    if t == 16:  # per source and launch: 112 + 48 chunk-end signals -> 7 + 3; RS2 48 -> 24 batches
        c0, c1 = SP.rs_stream_counts(16), SP.rs_stream_counts(16, rsb=True)
        assert [c0[k]["signals"] for k in c0] == [112, 48] and [c1[k]["signals"] for k in c1] == [7, 3]
        assert (c0["rs2"]["batches"], c1["rs2"]["batches"]) == (48, 24)
        assert [c1[k]["cb_bytes"] for k in c1] == [c0[k]["cb_bytes"] for k in c0] == [81920, 32768]
        # RSNP at t 16, per source and launch: RS1 560 -> 504 packets (84 batches of 12 in 3 slots), RS2 96 -> 72 (18
        # of 8); the column's middle link 5.0 -> 4.5 MiB per direction, the row's 512 -> 384 KiB; signals stay 7 + 3
        c2 = SP.rs_stream_counts(16, rsb=True, rsnp=True)
        assert [(c1[k]["packets"], c2[k]["packets"]) for k in c1] == [(560, 504), (96, 72)]
        assert [(c1[k]["batches"], c2[k]["batches"]) for k in c1] == [(112, 84), (24, 18)]
        assert [(c1[k]["mid_bytes"], c2[k]["mid_bytes"]) for k in c1] == [(5242880, 4718592), (524288, 393216)]
        assert [c2[k]["signals"] for k in c2] == [7, 3] and [c2[k]["cb_bytes"] for k in c2] == [73728, 32768]


@pytest.mark.parametrize("t", [1, 2, 3, 4, 8, 16, 32])
def test_block_audits(t):
    """tt/tp_dsa_shard.py's per-chip block audit for AG1 / AG2 (shard_plan.audit_ag_blocks: the streams' blocks from
    ag_stream_args via FabricStreamAG._block's arithmetic vs Geometry.block) passes for rev 65 / AGK / AGB on all 32
    chips; asking the plan without agb reproduces ad057da04d8's device-build failure (chip (0, 1) ag1 dst 8 vs 16 t).
    The same for RS1 / RS2 (audit_rs_blocks: FabricStreamScatter._block / _dsts from rs_stream_args) with RSB and RS1
    early on and off; a plan without rsb under RSB streams fails the audit (chip (0, 0) rs1's bases q vs q t)"""
    geo = SP.Geometry(8, 4)
    for agk, agb in ((False, False), (True, False), (True, True)):
        SP.audit_ag_blocks(geo, t, agk=agk, agb=agb)
        a = SP.ag_stream_args(t, agk=agk, agb=agb)
        st = geo.streams(t, agk=agk, agb=agb)
        assert (st["ag1"]["chunks"], st["ag2"]["chunks"]) == (a["ag1"]["rows"] // 32, a["ag2"]["rows"] // 32)
        assert st["ag2"]["chunks"] == geo.P * st["ag1"]["chunks"]
        # every block's pages: 32 blocks x (chunks x width / 32) = the format's pages per token x t
        assert st["ag1"]["chunks"] * st["ag1"]["width"] // 32 == (SP.AGK_PPC if agk else SP.AG32_PPC) * t
        assert geo.holders(0, 1, t) and geo.header_budget(0, 1, t)  # (links / headers: independent of the chunks)
    if t > 1:
        bad = f"chip (0, 1) stream ag1: blocks {{'p': 0, 'src': 0, 'dst': 8, 'local': True, 'up': 7, 'dn': 0, 'dsts': []}} differ from the plan {{'p': 0, 'src': 0, 'dst': {8 * t},"
        with expect_error(ValueError, re.escape(bad)):
            SP.audit_ag_blocks(geo, t, agk=True, agb=True, plan_knobs=dict(agk=True, agb=False))
    for rsb in (False, True):
        for early in (False, True):
            SP.audit_rs_blocks(geo, t, rsb=rsb, rs1_early=early)
        a, st = SP.rs_stream_args(t, rsb=rsb), geo.streams(t, rsb=rsb)
        for name, ppc in (("rs1", SP.RS2_PPC), ("rs2", SP.RS2B_PPC)):
            assert (st[name]["chunks"], st[name]["width"]) == (a[name]["rows"] // 32, a[name]["width"])
            assert st[name]["chunks"] * st[name]["width"] // 32 == ppc * t  # a destination block: t slices
        assert geo.holders(0, 1, t) == geo.holders(0, 1, 1) and geo.header_budget(0, 1, t) == geo.header_budget(0, 1, 1)
    a, an = SP.rs_stream_args(t, rsb=True), SP.rs_stream_args(t, rsb=True, rsnp=True)  # RSNP: RSB's rows / width
    assert all((an[k]["rows"], an[k]["width"]) == (a[k]["rows"], a[k]["width"]) for k in a)  # -> RSB's blocks / audit
    if t > 1:
        bad = "chip (0, 0) stream rs1: blocks {'p': 0, 'src': None, 'dst': 0, 'local': False, 'up': 7, 'dn': 0, 'dsts': [(1, 1, 0, 1),"
        with expect_error(ValueError, re.escape(bad) + f".*'dsts': \\[\\(1, {t}, 0, 1\\),"):
            SP.audit_rs_blocks(geo, t, rsb=True, plan_knobs=dict(rsb=False))
    with expect_error(AssertionError, "RSB: the 8x4 scatters"):
        SP.Geometry(2, 4).streams(t, rsb=True)


def test_agb_knob(monkeypatch):
    """GLM53_DSA_SHARD_AGB: default off, the 8x4 only, only on top of GLM53_DSA_SHARD_AGK"""
    for k in ("GLM53_DSA_SHARD_AGK", "GLM53_DSA_SHARD_AGB"):
        monkeypatch.delenv(k, raising=False)
    assert not SP.shard_agb(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_AGB", "1")
    with expect_error(ValueError, "GLM53_DSA_SHARD_AGB=1 needs GLM53_DSA_SHARD_AGK=1"):
        SP.shard_agb(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_AGK", "1")
    assert SP.shard_agb(8) and not SP.shard_agb(2)


# ------------------------------------------------------------------ the shard tail knobs compose; the 8x4 defaults
KNOB_ENV = {
    "comb": "GLM53_DSA_SHARD_COMB",
    "rs1_early": "GLM53_DSA_SHARD_RS1_EARLY",
    "agk": "GLM53_DSA_SHARD_AGK",
    "agb": "GLM53_DSA_SHARD_AGB",
    "rsb": "GLM53_DSA_SHARD_RSB",
    "rsnp": "GLM53_DSA_SHARD_RSNP",
}


def _knob_env(monkeypatch, on: dict) -> None:
    """exactly the knobs of `on` set to 1, the others absent"""
    for k, var in KNOB_ENV.items():
        if on.get(k):
            monkeypatch.setenv(var, "1")
        else:
            monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("t", [1, 4, 16])
@pytest.mark.parametrize("bits", range(1 << len(KNOB_ENV)))
def test_shard_knobs_compose(monkeypatch, bits, t):
    """every subset of COMB / RS1_EARLY / AGK / AGB / RSB / RSNP: each flag reads its own variable only (the 8x4 only;
    AGB without AGK and RSNP without RSB are refused), and every chip's stream plan takes RS1_EARLY's destination order,
    AGK / AGB's AG format and RSB's scatter chunks independently -- RS1 early reorders the column scatter's destinations
    and nothing else, the AG blocks follow the AG knobs only, the scatters' chunk bases follow RSB only (RSNP keeps
    RSB's rows / width: only the lanes' staged pages change), the streams' blocks == the plan's (audit_ag_blocks /
    audit_rs_blocks), link holders / packet headers are the knob-off plan's; AGK's key-packed y_ag keeps every page once
    and the q pieces where the merge / ix_dm1 read them"""
    on = {k: bool(bits >> i & 1) for i, k in enumerate(KNOB_ENV)}
    _knob_env(monkeypatch, on)
    if on["agb"] and not on["agk"]:
        with expect_error(ValueError, "GLM53_DSA_SHARD_AGB=1 needs GLM53_DSA_SHARD_AGK=1"):
            SP.shard_agb(8)
        return
    if on["rsnp"] and not on["rsb"]:
        with expect_error(ValueError, "GLM53_DSA_SHARD_RSNP=1 needs GLM53_DSA_SHARD_RSB=1"):
            SP.shard_rsnp(8)
        return
    got = dict(comb=SP.shard_comb(8), rs1_early=SP.shard_rs1_early(8), agk=SP.shard_agk(8), agb=SP.shard_agb(8),
               rsb=SP.shard_rsb(8), rsnp=SP.shard_rsnp(8))  # fmt: skip
    assert got == on
    assert not (SP.shard_comb(2) or SP.shard_rs1_early(2) or SP.shard_agk(2) or SP.shard_agb(2) or SP.shard_rsb(2))
    assert not SP.shard_rsnp(2)
    a, a0 = SP.rs_stream_args(t, rsb=on["rsb"], rsnp=on["rsnp"]), SP.rs_stream_args(t, rsb=on["rsb"])
    assert all((a[k]["rows"], a[k]["width"]) == (a0[k]["rows"], a0[k]["width"]) for k in a)
    assert all(("sub" in a[k]["extra"]) == on["rsnp"] for k in a)
    g, early, ag, rs = SP.Geometry(8, 4), on["rs1_early"], dict(agk=on["agk"], agb=on["agb"]), dict(rsb=on["rsb"])
    SP.audit_ag_blocks(g, t, **ag)
    SP.audit_rs_blocks(g, t, rs1_early=early, **rs)
    for r in range(8):
        for c in range(4):
            SP.audit_links(g.holders(r, c, t))
            SP.audit_headers(g.header_budget(r, c, t))
            for name in ("ag1", "ag2"):
                assert g.block(name, t, r, c, rs1_early=early, **ag, **rs) == g.block(name, t, r, c, **ag)
            assert g.block("rs2", t, r, c, rs1_early=early, **ag, **rs) == g.block("rs2", t, r, c, **rs)
            b, b0 = g.block("rs1", t, r, c, rs1_early=early, **ag, **rs), g.block("rs1", t, r, c, **rs)
            assert sorted(b["dsts"]) == sorted(b0["dsts"])
            assert {k: v for k, v in b.items() if k != "dsts"} == {k: v for k, v in b0.items() if k != "dsts"}
            assert [d[0] for d in b["dsts"][:4]] == SP.rs1_early_first(r, 8) if early else b["dsts"] == b0["dsts"]
            for name in ("rs1", "rs2"):  # RSB: the chunk bases count blocks (the knob-off's divided by t)
                k1, k0 = g.block(name, t, r, c, **rs), g.block(name, t, r, c)
                m = t if on["rsb"] else 1
                assert k1["dst"] * m == k0["dst"] and [(q, s_ * m, d, h) for q, s_, d, h in k1["dsts"]] == k0["dsts"]
    if on["agk"]:  # the packed y_ag: 32 blocks x t tokens x AGK_PPC pages, each page once; q pieces in page 0
        pages = {SP.agk_key_page(t, blk, tok) for blk in range(32) for tok in range(t)}
        pages |= {SP.agy_q_page(t, blk, tok, d)[0] for blk in range(32) for tok in range(t) for d in (0, 15)}
        assert sorted(pages) == list(range(32 * t * SP.AGK_PPC))


def _protect_env(monkeypatch, keys) -> None:
    """keys absent, and restored (absent or not) after the test: galaxy_defaults.apply writes os.environ directly"""
    for k in keys:
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)


def test_galaxy_defaults_shard_tail(monkeypatch):
    """tt/galaxy_defaults.py: the 8x4 default turns on AGK, AGB, RS1_EARLY, RSB, R8 and RSNP (validated: 4 K x 16
    bit-identical to knob-off over 3 runs; 32 K x 4 ids identical) and the fused prefill mHC mix, and leaves COMB (ids
    move at a near-tie) opt-in; an
    explicit environment value wins (an explicit AGK=0 drops the AGB default with it, an explicit RSB=0 the RSNP
    default); other meshes get nothing"""
    from models.autoports.zai_org_glm_5_3_flash.tt import galaxy_defaults as GD

    assert GD.KNOBS["GLM53_DSA_SHARD_AGK"] == "1"
    assert GD.KNOBS["GLM53_MKX_XROWS_COAL"] == "1"
    assert GD.KNOBS["GLM53_DSA_SHARD_RS1_EARLY"] == "1" and GD.KNOBS["GLM53_DSA_SHARD_AGB"] == "1"
    assert "GLM53_DSA_SHARD_COMB" not in GD.KNOBS
    assert GD.KNOBS["GLM53_DSA_SHARD_RSB"] == "1" and GD.KNOBS["GLM53_DSA_SHARD_R8"] == "1"
    assert GD.KNOBS["GLM53_DSA_SHARD_RSNP"] == "1"
    assert GD.KNOBS["GLM53_DSA_SHARD_LOCAL_ALL"] == "1"
    assert GD.KNOBS["GLM53_PF_MHC_POST"] == "fused"
    assert GD.KNOBS["GLM53_PF_MOE_KB"] == "808" and "GLM53_PF_MOE_FID" not in GD.KNOBS
    assert all(
        GD.KNOBS[k] == "1"
        for k in (
            "GLM53_PF_MOE_XGATHER",
            "GLM53_PF_MOE_V2",
            "GLM53_PF_MOE_RM_SKIP",
            "GLM53_PF_MOE_PADMASK",
            "GLM53_PF_MOE_XFFN_VALID",
            "GLM53_PF_MOE_XFFN_SNAKE",
        )
    )
    keys = list(GD.KNOBS) + ["GLM53_MODEL_TRACE_MB", "GLM53_PF_SPARSE_T", "GLM53_PREFILL_BUCKETS"]
    keys += [v for v in KNOB_ENV.values() if v not in keys] + ["GLM53_DSA_SHARD_AGB"]
    _protect_env(monkeypatch, keys)
    assert GD.apply((2, 4), 4096) == {} and not SP.shard_agk(8)
    got = GD.apply((8, 4), 4096, 4096)
    assert got["GLM53_DSA_SHARD_AGK"] == got["GLM53_DSA_SHARD_AGB"] == got["GLM53_DSA_SHARD_RS1_EARLY"] == "1"
    assert SP.shard_agk(8) and SP.shard_agb(8) and SP.shard_rs1_early(8) and not SP.shard_comb(8)
    assert got["GLM53_DSA_SHARD_RSB"] == "1" and SP.shard_rsb(8)
    assert got["GLM53_DSA_SHARD_R8"] == "1" and SP.shard_r8(8)
    assert got["GLM53_DSA_SHARD_RSNP"] == "1" and SP.shard_rsnp(8)
    assert got["GLM53_DSA_SHARD_LOCAL_ALL"] == "1" and SP.shard_local_all(8)
    assert got["GLM53_PF_MHC_POST"] == "fused"
    assert got["GLM53_PF_MOE_KB"] == "808" and "GLM53_PF_MOE_FID" not in got
    assert all(
        got[k] == "1"
        for k in (
            "GLM53_PF_MOE_XGATHER",
            "GLM53_PF_MOE_V2",
            "GLM53_PF_MOE_RM_SKIP",
            "GLM53_PF_MOE_PADMASK",
            "GLM53_PF_MOE_XFFN_VALID",
            "GLM53_PF_MOE_XFFN_SNAKE",
        )
    )
    _protect_env(monkeypatch, keys)
    monkeypatch.setenv("GLM53_DSA_SHARD_AGK", "0")  # an explicit value wins
    got = GD.apply((8, 4), 4096, 4096)
    assert "GLM53_DSA_SHARD_AGK" not in got and not SP.shard_agk(8)
    assert "GLM53_DSA_SHARD_AGB" not in got and not SP.shard_agb(8)  # dropped with AGK, no AGB-needs-AGK error
    assert got["GLM53_DSA_SHARD_RSNP"] == "1" and SP.shard_rsnp(8)  # (RSB still on)
    _protect_env(monkeypatch, keys)
    monkeypatch.setenv("GLM53_DSA_SHARD_RSB", "0")  # an explicit RSB=0 drops the RSNP default with it
    got = GD.apply((8, 4), 4096, 4096)
    assert "GLM53_DSA_SHARD_RSB" not in got and not SP.shard_rsb(8)
    assert "GLM53_DSA_SHARD_RSNP" not in got and not SP.shard_rsnp(8)  # no RSNP-needs-RSB error
    assert got["GLM53_DSA_SHARD_AGB"] == got["GLM53_DSA_SHARD_R8"] == "1"


@pytest.mark.parametrize("lanes", [1, 4, 8])
def test_sel2_matches_the_rounds(lanes):
    """rev 73 GLM53_DSA_SHARD_SEL2: the local select with packed report / pull sums, no per-round above counts (the
    cores count the keys their compactions drop upward and report them in the tie round) and per-owner quota chunks
    gives R8's prefixes, k_rem, both levels' sums per round, (quota, offset) pairs, header count, kept pairs and records
    bitwise -- the cores' upward drops equal the rounds' above counts -- for random, heavy-tie, all-equal (the bare
    probe's unwritten pool) and fp32-clustered keys, past 512 local pools up to 1 M's 8192 and at the edges"""
    gen = torch.Generator().manual_seed(73 + lanes)
    rnd = random.Random(73 + lanes)
    sizes = [0, 1, 511, 512, 513, 600, 1024, 2047, 4096, 8192] + rnd.sample(range(514, 8192), 10)
    for n in sizes:
        lq = torch.arange(n)
        for mode in ("random", "heavy", "equal", "clustered"):
            if mode == "random":
                keys = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
            elif mode == "heavy":
                keys = torch.randint(0, 4, (n,), generator=gen, dtype=torch.int64)
            elif mode == "equal":
                keys = torch.full((n,), 0x80000000, dtype=torch.int64)
            else:  # relu'd fp32 scores in a narrow band: the top bytes agree, the low ones decide
                keys = 0x3F000000 + torch.randint(0, 1 << 18, (n,), generator=gen, dtype=torch.int64)
            core_keys = SP.lane_core_keys(keys, lq, 64, lanes)
            want = SP.local_select(core_keys, 512, 8)
            got = SP.local_select_sel2(core_keys, 512)
            for key in ("T", "k_rem", "qpair", "n_rec", "kept", "records", "hists", "messages"):
                assert got[key] == want[key], (n, lanes, mode, key)
            assert got["gt"] == want["above"], (n, lanes, mode)
            assert SP.x_ag_chunk(got["records"], n_rec=got["n_rec"]) == SP.x_ag_chunk(
                want["records"], n_rec=want["n_rec"]
            )
    with expect_error(ValueError, "keys on the lane > 65535"):
        SP.local_select_sel2([[1] * 65536], 512)


def test_sel2_packed_sums_and_quota_chunks():
    """dsa_shard.hpp SEL2's packed sums are exact up to 65535 per count over the lane (64 cores x 1023 + 63 in one
    bin, 65535 from one core) and catch the carry past it; every owner's 16-B chunk holds its own (quota, offset) pair
    (sel2_quota_chunks: ix_dm0.cpp send_own) for odd / even core counts"""
    rnd = random.Random(7300)
    slots = [SP.sel2_pack([1023 + (c == 0) * 63] + [rnd.randrange(0, 64) for _ in range(15)]) for c in range(64)]
    out = SP.sel2_sum(slots)
    assert out[0] == 64 * 1023 + 63 == 65535
    assert SP.sel2_sum([SP.sel2_pack([65535] * 16)]) == [65535] * 16
    with expect_error(AssertionError, "carry between the halves"):
        SP.sel2_sum([SP.sel2_pack([65535] + [0] * 15), SP.sel2_pack([1] + [0] * 15)])
    with expect_error(AssertionError, ""):
        SP.sel2_pack([65536] + [0] * 15)
    for C in (1, 2, 7, 8, 16, 63, 64):
        qp = [(rnd.randrange(0, 600), rnd.randrange(0, 8192)) for _ in range(C)]
        assert SP.sel2_quota_chunks(qp) == qp, C


def test_sel2_knobs(monkeypatch):
    """GLM53_DSA_SHARD_SEL2: default off, the 8x4 only, 0 / 1 only, needs R8; the kernels and mk_mla.py carry the same
    define / helper names (a renamed define would silently build knob-off)"""
    monkeypatch.delenv("GLM53_DSA_SHARD_SEL2", raising=False)
    assert not SP.shard_sel2(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_SEL2", "1")
    assert SP.shard_sel2(8, r8=True) and not SP.shard_sel2(2, r8=False)
    with expect_error(ValueError, "needs GLM53_DSA_SHARD_R8=1"):
        SP.shard_sel2(8, r8=False)
    monkeypatch.setenv("GLM53_DSA_SHARD_SEL2", "on")
    with expect_error(ValueError, "GLM53_DSA_SHARD_SEL2=on: 0 or 1"):
        SP.shard_sel2(8)
    import os

    kd = os.path.join(SP.KDIR, "mla")
    for f, names in (
        ("ix_dm0.cpp", ("MLA_DSA_SHARD_SEL2", "sel2_sum", "send_own", "sel2_fine")),
        ("att_dsa_shard_dm1.cpp", ("MLA_DSA_SHARD_SEL2", "r8_hist_packed", "hist[1] = gt")),
        ("dsa_shard.hpp", ("MLA_DSA_SHARD_SEL2", "r8_hist_packed", "sel2_sum")),
    ):
        with open(os.path.join(kd, f)) as fh:
            src = fh.read()
        assert all(nm in src for nm in names), (f, names)
    with open(os.path.join(os.path.dirname(SP.KDIR), "mk_mla.py")) as fh:
        src = fh.read()
    assert '("MLA_DSA_SHARD_SEL2", "1")' in src


def test_zz_no_ttnn_was_imported_by_the_suite():
    """The unit CI runs this file without the root conftest: a host suite must never pull ttnn in (it would open
    every chip of the box beside a running mesh job)."""
    assert "ttnn" not in sys.modules
