# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Packed prefill, the CPU rung (PB.1; bringup/PACKED_DESIGN.md "Test plan" rung 1). No device.

1. The descriptor and the per-segment index math (tt/packed.py): today's single header as a descriptor, lane packing /
   unpacking, per-row positions, valid / write rows (PREFILL_END=cap, global_write_start), the block-diagonal causal
   mask, lane -> chip maps and window-carry sources on every box at 8192, compressor groups at ratios 1 / 2 / 4 / 128,
   window carry classes, engram tails, the snapshot range, the DRAM budget at 1M.
2. The CPU reference run packed vs alone (reference/packed_ref.py over reference/ds41_ref.py; the real checkpoint's
   layers 0-3: window-only 0, engram 1, KV + index source 2 with ratio-2 groups, a reader of 2's top-k 3). A chunk of
   4 lanes x 64 rows (the 4 x 2048 geometry scaled down), five slots with short histories of their own (different
   texts, positions 0..160), one fresh slot, padded and odd-length segments (a ratio-2 group left pending), an empty
   lane:
     lane equivalence   each segment packed with 3 others == the segment alone in the same geometry (other lanes empty),
                        every block boundary of every layer (its rows) and final_norm, bit for bit; and packed again in a
                        different lane with different neighbours (and an empty lane) == the first packing
     alone == today     each segment's packed rows == today's single header {slot, actual_start, actual_end} on the
                        same chunk (ChunkDescriptor.single: one segment over all CHUNK rows), bit for bit, state too
     vs forward_chunk   DS41Reference.forward_chunk of the segment by itself (a chunk of its own rows: the reference's
                        dense GEMMs see other row counts, ~1e-6 rel): index states (top-k, engram tail) equal, float
                        states written at the same rows; the rel gap recorded (diagnostic)
     slot digests       every state tensor of every slot after the packed chunk == after the segment alone
     determinism        the same packed chunk twice: identical
   Every side runs the routed experts' SwiGLU row by row (reference/packed_ref.fixed_expert_tiles: GEMMs in fixed
   32-row tiles, elementwise steps one row at a time): the CPU reference's grouped SwiGLU is not composition-independent
   (torch's multithreaded elementwise kernels: 1 ulp at L3.experts_out on this packing, DS41-321); the device MoE's own
   row-locality is the device oracle's (test_packed_oracle.py, PB.3 / PB.4).
Recorded: packed_cpu_cases, packed_cpu_failures (gated: >= 1, == 0); packed_cpu_lane_equiv_max_abs_diff,
packed_cpu_alone_vs_today_max_abs_diff, packed_cpu_slot_digest_mismatches, packed_cpu_determinism_max_abs_diff;
packed_cpu_vs_forward_chunk_rel (diagnostic).

    PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.1 --spec $BRINGUP_SPEC --check
(the gate runs ``python -m pytest -q`` on this file: CPU only, ~3-4 min, 4 layers of the real checkpoint on the host).
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.deepseek_v41_flash_d_p.tt import packed as P

RESULTS = {
    "cases": 0,
    "failures": [],
    "lane_equiv": 0.0,
    "alone_today": 0.0,
    "digest": 0,
    "determinism": 0.0,
    "vs_forward_chunk_rel": 0.0,
}
LAYERS = [0, 1, 2, 3]
LANE = 64  # lane rows of the CPU rung (2048 on the device)
CHUNK = 4 * LANE
BOXES = {"galaxy_8x4": (8, 4), "galaxy_4x4": (4, 4), "loudbox_2x4": (2, 4), "quietbox_2x2": (2, 2)}


@contextmanager
def case(name: str):
    RESULTS["cases"] += 1
    try:
        yield
    except Exception as e:  # (counted, then re-raised: pytest shows it)
        RESULTS["failures"].append(f"{name}: {type(e).__name__}: {e}")
        raise


@pytest.fixture(scope="module", autouse=True)
def _record():
    yield
    metrics.record("packed_cpu_cases", RESULTS["cases"])
    metrics.record("packed_cpu_failures", len(RESULTS["failures"]))
    metrics.record("packed_cpu_lane_equiv_max_abs_diff", RESULTS["lane_equiv"])
    metrics.record("packed_cpu_alone_vs_today_max_abs_diff", RESULTS["alone_today"])
    metrics.record("packed_cpu_slot_digest_mismatches", RESULTS["digest"])
    metrics.record("packed_cpu_determinism_max_abs_diff", RESULTS["determinism"])
    metrics.record("packed_cpu_vs_forward_chunk_rel", RESULTS["vs_forward_chunk_rel"])
    print(f"packed CPU rung: {RESULTS['cases']} cases, {len(RESULTS['failures'])} failures")
    for f in RESULTS["failures"]:
        print(f"  FAIL {f}")


# ============================================================================================= 1. index math
def test_single_header_is_todays_chunk():
    with case("single header"):
        for start, end, gws in [(0, 8192, None), (139264, 141312, None), (51200, 56319, None), (4096, 9000, 4224)]:
            d = P.ChunkDescriptor.single(8192, 3, start, end, gws)
            assert d.lanes == 1 and d.lane_rows == 8192
            assert torch.equal(d.row_positions(), start + torch.arange(8192))
            cap = end // 64 * 64
            v = d.row_valid("cap")
            assert int(v.sum()) == cap - start and bool(v[: cap - start].all())
            assert int(d.row_valid("end").sum()) == end - start
            lo = start if gws is None else gws
            w = d.row_writes("cap")
            assert int(w.sum()) == max(0, cap - lo) and (lo == start or not bool(w[: lo - start].any()))
            assert d.row_slot().unique().tolist() == [3]
            assert d.segments[0].replay == (gws is not None and gws > start)


def test_lanes_positions_masks(expect_error):
    with case("lanes"):
        segs = [P.Segment(5, 139264, 141312), P.Segment(2, 61440, 62000), None, P.Segment(9, 0, 37)]
        d = P.ChunkDescriptor.packed(8192, segs)
        assert d.lanes == 4 and d.lane_rows == 2048 and d.new_tokens() == 2048 + 560 + 37
        pos, lane, slot = d.row_positions(), d.row_lane(), d.row_slot()
        for i, s in enumerate(segs):
            a = 2048 * i
            assert bool((lane[a : a + 2048] == i).all())
            want = torch.arange(2048) + (0 if s is None else s.start)
            assert torch.equal(pos[a : a + 2048], want), i
            assert bool((slot[a : a + 2048] == (-1 if s is None else s.slot)).all())
        v = d.row_valid("end")
        assert int(v.sum()) == d.new_tokens() and not bool(v[4096:6144].any())
        vc = d.row_valid("cap")  # 62000 -> cap 61952: 512 rows; 37 -> cap 0: nothing computed in lane 3
        assert int(vc[2048:4096].sum()) == 512 and not bool(vc[6144:].any())
    with case("causal mask block-diagonal"):
        segs = [P.Segment(1, 96, 96 + 40), P.Segment(2, 0, 64), None, P.Segment(4, 32, 32 + 17)]
        d = P.ChunkDescriptor.packed(256, segs)
        m = d.causal_mask("end")
        lane = d.row_lane()
        assert not bool((m & (lane.unsqueeze(1) != lane.unsqueeze(0))).any()), "a row reads another lane"
        for i, s in enumerate(segs):
            blk = m[64 * i : 64 * i + 64, 64 * i : 64 * i + 64]
            n = 0 if s is None else s.length
            want = torch.tril(torch.ones(64, 64, dtype=torch.bool))
            want[:, n:] = False
            assert torch.equal(blk, want), i
    with case("ids pack / unpack"):
        segs = [P.Segment(1, 96, 136), None, P.Segment(2, 0, 64), P.Segment(4, 32, 49)]
        d = P.ChunkDescriptor.packed(256, segs)
        ids = [torch.arange(40) + 1000, None, torch.arange(64) + 2000, torch.arange(17) + 3000]
        t = d.pack_ids(ids)
        assert int((t == P.PAD_ID).sum()) == 256 - 40 - 64 - 17
        back = d.unpack_rows(t)
        assert back[1] is None
        for i in (0, 2, 3):
            assert torch.equal(back[i], ids[i])
    with case("descriptor refuses"):
        for bad, msg in (
            (lambda: P.Segment(0, 100, 200), "segment start 100 must be 32-aligned"),
            (lambda: P.ChunkDescriptor.packed(8192, [P.Segment(0, 0, 4096)]), "segment of 4096 rows > lane of 2048"),
            (
                lambda: P.ChunkDescriptor.packed(8192, [P.Segment(0, 0, 64), P.Segment(0, 64, 128)]),
                "two segments of one slot in a chunk",
            ),
        ):
            with expect_error(AssertionError, msg):
                bad()


def test_mesh_maps_every_box(expect_error):
    with case("lane chips / carry sources"):
        for name, (sp, tp) in BOXES.items():
            d = P.ChunkDescriptor.packed(8192, [P.Segment(7, 0, 2048), None, P.Segment(3, 1024, 2000), None])
            lc = d.lane_chips(sp, tp)
            n = sp * tp
            assert sorted(q for c in lc for q in c) == list(range(n)), name
            per = 8192 // n
            for i, chips in enumerate(lc):  # chip q holds natural rows [q per, (q + 1) per)
                assert chips[0] * per == 2048 * i and (chips[-1] + 1) * per == 2048 * (i + 1), (name, i)
            src = d.carry_sources(sp, tp)
            for i, chips in enumerate(lc):
                assert src[chips[0]] == ("slot", i), (name, i)
                assert all(src[q] == ("prev", q - 1) for q in chips[1:]), (name, i)
            hs = d.history_slots(sp, tp)
            assert [hs[c[0]] for c in lc] == [7, -1, 3, -1], name
        # 3 lanes of 2048 over 8x4's 192-row chips: a lane is not whole chips
        with expect_error(AssertionError, "lane of 2048 rows is not whole chips of 192 rows"):
            P.ChunkDescriptor(6144, (P.Segment(0, 0, 32), None, None)).lane_chips(8, 4)
        with expect_error(AssertionError, "chunk 1536 does not split into tiles over 32 chips"):  # DS41-297
            P.ChunkDescriptor(1536, (P.Segment(0, 0, 32),)).lane_chips(8, 4)


def test_compressor_groups():
    with case("compressor groups"):
        for r in (1, 2, 4, 128):
            for start in (0, 32, 96, 4096, 139264 + 32):
                for n in (1, 17, 64, 2047, 2048):
                    for gws in (None, start + 64):
                        s = P.Segment(0, start, start + n, gws)
                        g = P.compress_groups(s, r, "end")
                        lo, hi = P.write_range(s, "end")
                        brute = [j for j in range(0, (start + n) // r + 1) if j * r >= lo and (j + 1) * r <= hi]
                        want = (brute[0], brute[-1] + 1) if brute else None
                        got = g["groups"] if g["groups"][1] > g["groups"][0] else None
                        assert got == want, (r, start, n, gws, got, want)
                        assert g["carry_in"] == start % r and g["carry_out"] == (start + n) % r
                        if r in (1, 2):  # this checkpoint's ratios: 32-aligned starts never carry a group in
                            assert g["carry_in"] == 0
        assert P.compress_groups(P.Segment(0, 0, 10), 0)["g0"] is None


def test_window_engram_snapshot():
    with case("window carry"):
        assert P.window_carry(P.Segment(0, 0, 10)) == {"lo": 0, "hi": 0, "cls": 0}
        assert P.window_carry(P.Segment(0, 64, 100)) == {"lo": 0, "hi": 64, "cls": 64}
        assert P.window_carry(P.Segment(0, 1024, 3000)) == {"lo": 896, "hi": 1024, "cls": 128}
        assert P.window_carry(P.Segment(0, 1024, 3000, 1152))["cls"] == 0  # replay: truncated at start
    with case("engram tail"):
        hist = torch.arange(100, 200)
        assert P.engram_tail(hist, 0).tolist() == [-1, -1, -1]
        assert P.engram_tail(hist, 2).tolist() == [-1, 100, 101]
        assert P.engram_tail(hist, 64).tolist() == [161, 162, 163]
    with case("snapshot / write range"):
        s = P.Segment(0, 4096, 6000)
        assert P.snapshot_range(s) == (5952 - 128, 5952)  # cap(6000) = 5952 (KV block 64)
        assert P.write_range(s) == (4096, 5952) and P.write_range(s, "end") == (4096, 6000)
        assert P.write_range(P.Segment(0, 4096, 6000, 8192)) == (5952, 5952)  # all below the frontier: nothing


def test_dram_budget_1m():
    with case("dram budget"):
        h = [1044480 - 2048] * 4
        v1 = P.history_bytes_per_chip(h)
        v2 = P.history_bytes_per_chip(h, per_lane=True)
        one = P.history_bytes_per_chip(h[:1])
        assert v1["table"] == 4 * one["table"] and v2["table"] == one["table"]
        gb = 1e9
        print(
            f"history buffers a chip at 1M: today (1 segment) {one['total'] / gb:.3f} GB; packed v1 (4 tables on every "
            f"chip) {v1['total'] / gb:.3f} GB; packed v2 (each chip its lane's table) {v2['total'] / gb:.3f} GB"
        )
        assert v1["total"] < 4 * gb  # (32 GB a chip; ~9 GiB used at 1M today, P.87)


# ============================================================================================= 2. reference packed vs alone
def _spec():
    from models.demos.common.bringup.reference.golden import load_spec

    if not os.environ.get("BRINGUP_SPEC"):
        pytest.skip("BRINGUP_SPEC not set")
    return load_spec(None)


# five sessions: (slot, history tokens, new tokens, text offset); starts are the histories (32-aligned)
SESSIONS = {
    "A": (3, 160, 64, 0),
    "B": (0, 96, 40, 700),
    "C": (7, 0, 64, 1400),
    "D": (1, 64, 37, 2100),
    "E": (5, 128, 51, 2800),
}
COMP1 = ["A", "B", "C", "D"]
COMP2 = ["C", "A", "E", None]


@pytest.fixture(scope="module")
def rig():
    from models.demos.common.bringup.core.metrics import cpu_threads
    from models.demos.common.bringup.reference.golden import text_tokens

    torch.set_num_threads(cpu_threads())
    spec = _spec()
    ref = spec.hooks().reference(spec, layers=LAYERS, numerics="native")
    text = text_tokens(spec, 4096).long()
    max_seq = 512
    sess, base = {}, {}
    for k, (slot, hist, n, off) in SESSIONS.items():
        ids = text[off : off + hist + n]
        st = ref.new_state(max_seq)
        if hist:
            ref.forward_chunk(ids[:hist], 0, st)
        sess[k] = (slot, hist, ids[hist:])
        base[k] = st
    return ref, sess, base


def _desc(sess, comp):
    segs = [None if k is None else P.Segment(sess[k][0], sess[k][1], sess[k][1] + len(sess[k][2])) for k in comp]
    return P.ChunkDescriptor.packed(CHUNK, segs)


def _run_packed(ref, sess, base, comp, single: bool = False):
    from models.demos.deepseek_v41_flash_d_p.reference.packed_ref import forward_packed

    if single:  # today's header {slot, actual_start, actual_end}: one segment over the whole chunk
        (k,) = comp
        d = P.ChunkDescriptor.single(CHUNK, sess[k][0], sess[k][1], sess[k][1] + len(sess[k][2]))
    else:
        d = _desc(sess, comp)
    states = {sess[k][0]: copy.deepcopy(base[k]) for k in comp if k is not None}
    taps = {}
    out = forward_packed(
        ref,
        d,
        [None if k is None else sess[k][2] for k in comp],
        states,
        rec=lambda n, t: taps.__setitem__(n, t.clone()),
    )
    per = {}
    for lane, k in enumerate(comp):
        if k is None:
            continue
        a = d.lane_off(lane)
        b = a + len(sess[k][2])
        per[k] = {n: t[a:b] for n, t in taps.items()}
    return per, {k: states[sess[k][0]] for k in comp if k is not None}, out


def _run_today(ref, sess, base, k):
    slot, start, ids = sess[k]
    st = copy.deepcopy(base[k])
    taps = {}
    ref.forward_chunk(ids, start, st, rec=lambda n, t: taps.__setitem__(n, t.clone()))
    return taps, st


def _maxdiff(a: dict, b: dict, names=None) -> tuple[float, str]:
    worst, where = 0.0, ""
    for n in names or sorted(set(a) & set(b)):
        x, y = a[n], b[n]
        if x.shape != y.shape:
            return float("inf"), f"{n} shape {tuple(x.shape)} vs {tuple(y.shape)}"
        d = float((x.double() - y.double()).abs().max()) if x.numel() else 0.0
        if d > worst or (d != d):
            worst, where = d, n
    return worst, where


def _state_mismatches(ref, s1, s2) -> list[str]:
    bad = []
    for i in sorted(s1.window):
        for n in ("window", "ckv", "index_k", "partial", "topk", "cand", "tail"):
            a, b = getattr(s1, n).get(i), getattr(s2, n).get(i)
            if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                bad.append(f"L{i}.{n}")
    return bad


BOUNDARIES = None  # every boundary both runs recorded (block steps' outputs of every layer, final_norm)


# CPU only: 4 layers of the real checkpoint (~65 s load) + ~16 forwards (~240 s); above the default 300 s
@pytest.mark.timeout(1200)
def test_reference_packed_vs_alone(rig):
    from models.demos.deepseek_v41_flash_d_p.reference.packed_ref import fixed_expert_tiles

    with fixed_expert_tiles():  # (every side: the reference's grouped SwiGLU is not row-local, DS41-321)
        _packed_vs_alone(*rig)


def _packed_vs_alone(ref, sess, base):
    p1, st1, _ = _run_packed(ref, sess, base, COMP1)
    # every boundary the packed run records as rows (per-segment non-row outputs are not tapped)
    with case("determinism"):
        p1b, st1b, _ = _run_packed(ref, sess, base, COMP1)
        for k in COMP1:
            d, where = _maxdiff(p1[k], p1b[k])
            RESULTS["determinism"] = max(RESULTS["determinism"], d)
            assert d == 0.0, f"{k}: repeat differs at {where} by {d}"
            assert not _state_mismatches(ref, st1[k], st1b[k]), k
    for k in COMP1:  # lane equivalence: alone in the same geometry (other lanes empty)
        with case(f"lane equivalence {k}"):
            alone = [x if x == k else None for x in COMP1]
            pa, sa, _ = _run_packed(ref, sess, base, alone)
            d, where = _maxdiff(p1[k], pa[k])
            RESULTS["lane_equiv"] = max(RESULTS["lane_equiv"], d)
            bad = _state_mismatches(ref, st1[k], sa[k])
            RESULTS["digest"] += len(bad)
            assert d == 0.0, f"{k}: packed vs alone differs at {where} by {d}"
            assert not bad, f"{k}: slot state differs: {bad}"
    p2, st2, _ = _run_packed(ref, sess, base, COMP2)
    for k in ("A", "C"):  # another lane, other neighbours, an empty lane
        with case(f"lane move {k}"):
            d, where = _maxdiff(p1[k], p2[k])
            RESULTS["lane_equiv"] = max(RESULTS["lane_equiv"], d)
            bad = _state_mismatches(ref, st1[k], st2[k])
            RESULTS["digest"] += len(bad)
            assert d == 0.0 and not bad, f"{k}: lane {COMP1.index(k)} vs {COMP2.index(k)}: {where} {d}; {bad}"
    for k in list(SESSIONS):  # alone == today: today's single header on the same chunk (one lane of CHUNK rows)
        with case(f"alone == today {k}"):
            got, sp = (p1[k], st1[k]) if k in COMP1 else (p2[k], st2[k])
            ps, ss, _ = _run_packed(ref, sess, base, [k], single=True)
            d, where = _maxdiff(got, ps[k])
            RESULTS["alone_today"] = max(RESULTS["alone_today"], d)
            bad = _state_mismatches(ref, sp, ss[k])
            RESULTS["digest"] += len(bad)
            assert d == 0.0, f"{k}: packed lane vs today's single header differs at {where} by {d}"
            assert not bad, f"{k}: slot state vs today's single header: {bad}"
        with case(f"vs forward_chunk {k}"):  # the reference's own chunk of the segment (other GEMM row counts)
            taps, st = _run_today(ref, sess, base, k)
            names = [n for n in got if n in taps]
            assert any(n.endswith(".out") for n in names) and "final_norm" in names, sorted(got)[:5]
            r = max(_rel(got[n], taps[n]) for n in names)
            RESULTS["vs_forward_chunk_rel"] = max(RESULTS["vs_forward_chunk_rel"], r)
            bad = _structure_mismatches(sp, st)
            print(f"{k}: packed vs forward_chunk max rel {r:.3g} (diagnostic: GEMM row counts differ), structure {bad}")
            assert not bad, f"{k}: slot state structure vs forward_chunk: {bad}"


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def _structure_mismatches(s1, s2) -> list[str]:
    """Index states equal; float states written at the same rows (the same support)."""
    bad = []
    for i in sorted(s1.window):
        for n in ("topk", "cand", "tail"):
            a, b = getattr(s1, n).get(i), getattr(s2, n).get(i)
            if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                bad.append(f"L{i}.{n}")
        for n in ("window", "ckv", "index_k", "partial"):
            a, b = getattr(s1, n).get(i), getattr(s2, n).get(i)
            if (a is None) != (b is None) or (a is not None and not torch.equal(a.ne(0).any(-1), b.ne(0).any(-1))):
                bad.append(f"L{i}.{n} rows")
    return bad
