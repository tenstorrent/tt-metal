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
    gen = torch.Generator().manual_seed(7)
    n_pools = 32 * C * 6  # ~6 pool pages per chip
    keys = torch.randint(1, 1 << 31, (n_pools,), generator=gen, dtype=torch.int64)
    res = SP.emulate_lane_merge(keys, C, chunks=64, lanes=lanes, k=512)
    ref = SP.serial_merge(res["lists"], 0, k=512)
    assert res["T_g"] == ref["T_g"], (res["T_g"], ref["T_g"])
    assert res["core_quota"][0] == ref["core_quota"]
    assert SP.lanes_reasons(16, 2, 64, 4, 1, nchips=C) == []
    assert SP.lanes_reasons(32, 4, 64, 4, 1, nchips=32) == ["32 merge helpers among 16 lane chunk cores"]


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
        else:
            raise ValueError(case)
        cores = torch.sort(torch.randint(0, 32, (m,), generator=gen)).values
        if case == "ties":
            keys = torch.randint(1, 40, (m,), generator=gen)
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


@pytest.mark.parametrize("case", ["random", "ties", "few", "empty", "one_chip", "all_empty"])
def test_helper_merge_kph(case):
    """the KPH merge (helper h holds chips h kph .. : one histogram over the union, in-place compaction per list,
    per-list tie counts) == the serial merge, and == kph 1 bitwise: the coordinator's round sums, T_g, the per-chip
    quotas, every chip's per-core quotas"""
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
    if case in ("few", "all_empty"):
        assert base["done"] and base["T_g"] == 0
    if case == "ties":
        assert not base["done"] and sum(base["helper_quota"]) == base["k_rem"] > 0  # a real tie round
    if case == "one_chip":
        assert sum(base["helper_quota"][q] for q in range(32) if q != 7) == 0


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


def test_rev65_knobs(monkeypatch):
    """GLM53_DSA_SHARD_COMB / GLM53_DSA_SHARD_RS1_EARLY: default off, the 8x4 only, 0 / 1 only; the fold's message
    size is the header's"""
    for k in ("GLM53_DSA_SHARD_COMB", "GLM53_DSA_SHARD_RS1_EARLY"):
        monkeypatch.delenv(k, raising=False)
    assert not SP.shard_comb(8) and not SP.shard_rs1_early(8)
    monkeypatch.setenv("GLM53_DSA_SHARD_COMB", "1")
    monkeypatch.setenv("GLM53_DSA_SHARD_RS1_EARLY", "1")
    assert SP.shard_comb(8) and SP.shard_rs1_early(8)
    assert not SP.shard_comb(2) and not SP.shard_rs1_early(2)  # the pair / the 2x4 keep rev 64's kernels
    monkeypatch.setenv("GLM53_DSA_SHARD_COMB", "2")
    with expect_error(ValueError, "GLM53_DSA_SHARD_COMB=2: 0 or 1"):
        SP.shard_comb(8)
    assert SP.HDR["COMB_MSG"] == 8


def test_zz_no_ttnn_was_imported_by_the_suite():
    """The unit CI runs this file without the root conftest: a host suite must never pull ttnn in (it would open
    every chip of the box beside a running mesh job)."""
    assert "ttnn" not in sys.modules
