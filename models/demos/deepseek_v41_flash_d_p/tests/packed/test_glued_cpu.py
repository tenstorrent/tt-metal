# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7 (owner 2026-11-20): glued packed prefill, the CPU rung. No device.

1. The glued descriptor (tt/packed.GluedChunk): engine rows vs model rows (PREFILL_END=cap drops each segment's
   [cap, end)), per-row positions / segments / writes, the block-diagonal causal mask, per-chip segments on every box,
   the split rule (a request that does not fit ends on a kv block and continues in a later chunk), the lane form
   (as_lanes) of a 4 x 2048 chunk, up to 6 segments.
2. The CPU reference run glued vs alone (reference/packed_ref.forward_packed runs a GluedChunk as is; real checkpoint
   layers 0-3; a 256-row chunk, segments of 37..90 rows back to back, model rows not tile-aligned here: the reference
   has no tiles):
     glued equivalence   each segment glued with others at other row offsets == the segment alone at row 0, every
                         block boundary and final_norm, bit for bit; slot state too
     alone == today      the segment alone == today's single header (ChunkDescriptor.single), bit for bit
     split == alone      a request split across two glued chunks (its first part at the end of chunk 1, the rest at
                         the start of chunk 2 next to another segment) == the same request chunked at the same
                         boundary alone
     determinism         the same glued chunk twice
   Every side runs the routed SwiGLU row by row (packed_ref.fixed_expert_tiles, DS41-321).

    PYTHONPATH=$PWD BRINGUP_SPEC=... python -m pytest -q models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_cpu.py
"""

from __future__ import annotations

import copy
import os

import pytest
import torch

from models.demos.deepseek_v41_flash_d_p.tt import packed as P

BOXES = {"galaxy_8x4": (8, 4), "galaxy_4x4": (4, 4), "loudbox_2x4": (2, 4), "quietbox_2x2": (2, 2)}


# ============================================================================================= 1. descriptor
def test_glued_rows_and_split(expect_error):
    segs = (P.Segment(1, 139264, 139264 + 1600), P.Segment(2, 4096, 4096 + 1650), P.Segment(3, 0, 777))
    g = P.GluedChunk(8192, segs)
    # engine rows back to back; model rows = [start, cap) back to back (cap: the kv block at or below end)
    assert [g.in_off(k) for k in range(3)] == [0, 1600, 3250]
    assert [g.rows(k) for k in range(3)] == [1600, 1600, 768]
    assert [g.lane_off(k) for k in range(3)] == [0, 1600, 3200] and g.used_rows() == 3968
    assert g.tile_aligned() and g.as_lanes() is None
    seg = g.row_seg()
    assert int((seg == 1).sum()) == 1600 and int(seg[3968]) == -1
    pos = g.row_positions()
    assert int(pos[1600]) == 4096 and int(pos[3200 + 767]) == 767
    eng = torch.arange(8192)
    mid = g.model_ids(eng)
    assert int(mid[1600]) == 1600 and int(mid[3200]) == 3250 and int(mid[3968]) == P.PAD_ID
    m = g.causal_mask()
    assert bool(m[1601, 1600]) and not bool(m[1600, 1601]) and not bool(m[1600, 1599]) and not bool(m[4000, 4000])
    # per chip segments: segments share chips (8x4: 256 rows a chip)
    cs = g.chip_segments(8, 4)
    assert cs[6] == [0, 1] and cs[12] == [1, 2] and cs[15] == [2] and cs[16] == []
    for box, (sp, tp) in BOXES.items():
        cs = g.chip_segments(sp, tp)
        assert sum(len(c) > 1 for c in cs) >= 1 or box == "quietbox_2x2", box
    n, exact = g.moe_real_prefix(8)
    assert exact and n == [1024, 1024, 1024, 896, 0, 0, 0, 0]
    # split: the first part ends on the last kv block that fits, the rest continues
    reqs = [P.Segment(k, 139264, 139264 + 2000) for k in range(5)]
    c1, rest = P.GluedChunk.fill(8192, reqs)
    assert c1.lanes == 5 and rest[0].slot == 4 and c1.segments[4].end == rest[0].start
    assert c1.segments[4].end % 64 == 0 and 8192 - c1.new_tokens() < 64  # (engine rows; model rows drop [cap, end))
    assert c1.used_rows() == 4 * 1984 + 192
    c2, rest2 = P.GluedChunk.fill(8192, rest)
    assert rest2 == [] and c2.segments[0].start == c1.segments[4].end and c2.segments[0].end == 139264 + 2000
    # 6 segments at most
    c6, r6 = P.GluedChunk.fill(8192, [P.Segment(k, 4096, 4096 + 640) for k in range(9)])
    assert c6.lanes == 6 and len(r6) == 3
    with expect_error(AssertionError, "a glued chunk holds 1..6 segments, got 7"):
        P.GluedChunk(8192, tuple(P.Segment(k, 0, 64) for k in range(7)))
    # a 4 x 2048 glued chunk is the lane chunk (the accepted lane path)
    g4 = P.GluedChunk(8192, tuple(P.Segment(k, 139264, 139264 + 2048) for k in range(4)))
    ln = g4.as_lanes()
    assert ln is not None and ln.lanes == 4 and torch.equal(ln.row_positions(), g4.row_positions())
    assert torch.equal(
        ln.pack_ids([torch.arange(2048) + k for k in range(4)]), g4.pack_ids([torch.arange(2048) + k for k in range(4)])
    )
    # a short last segment still is a lane chunk (its lane padded at the tail)
    g4s = P.GluedChunk(8192, tuple(P.Segment(k, 139264, 139264 + (2048 if k < 3 else 1600)) for k in range(4)))
    assert g4s.as_lanes() is not None
    # cap <= start: a segment that computes nothing has no model rows
    gq = P.GluedChunk(8192, (P.Segment(1, 2048, 2048 + 40), P.Segment(2, 0, 64)))
    assert gq.rows(0) == 0 and gq.lane_off(1) == 0


# ============================================================================================= 2. reference glued vs alone
def _spec():
    from models.demos.common.bringup.reference.golden import load_spec

    if not os.environ.get("BRINGUP_SPEC"):
        pytest.skip("BRINGUP_SPEC not set")
    return load_spec(None)


LAYERS = [0, 1, 2, 3]
CHUNK = 256
# (slot, history, new tokens, text offset); starts 32-aligned
SESSIONS = {
    "A": (3, 160, 64, 0),
    "B": (0, 96, 40, 700),
    "C": (7, 0, 64, 1400),
    "D": (1, 64, 37, 2100),
    "E": (5, 128, 90, 2800),
}
GLUE1 = ["A", "B", "C", "D"]  # 205 rows
GLUE2 = ["D", "E", "A"]  # 191 rows: A, D at other offsets, other neighbours


@pytest.fixture(scope="module")
def rig():
    from models.demos.common.bringup.core.metrics import cpu_threads
    from models.demos.common.bringup.reference.golden import text_tokens

    torch.set_num_threads(cpu_threads())
    spec = _spec()
    ref = spec.hooks().reference(spec, layers=LAYERS, numerics="native")
    text = text_tokens(spec, 4096).long()
    sess, base = {}, {}
    for k, (slot, hist, n, off) in SESSIONS.items():
        ids = text[off : off + hist + n]
        st = ref.new_state(512)
        if hist:
            ref.forward_chunk(ids[:hist], 0, st)
        sess[k] = (slot, hist, ids[hist:])
        base[k] = st
    return ref, sess, base, text


def _glued(ref, segs: list, base_states: dict):
    """One glued chunk of ``segs`` [(name, Segment, ids)] on copies of ``base_states`` -> ({name: taps of its rows},
    {name: state})."""
    from models.demos.deepseek_v41_flash_d_p.reference.packed_ref import forward_packed

    d = P.GluedChunk(CHUNK, tuple(s for _, s, _ in segs), prefill_end="end")
    states = {s.slot: base_states[n] for n, s, _ in segs}
    taps = {}
    forward_packed(ref, d, [ids for *_, ids in segs], states, rec=lambda n, t: taps.__setitem__(n, t.clone()))
    per = {}
    for k, (n, s, _) in enumerate(segs):
        a = d.lane_off(k)
        per[n] = {b: t[a : a + s.length] for b, t in taps.items()}
    return per, {n: states[s.slot] for n, s, _ in segs}


def _today(ref, seg, ids, st):
    from models.demos.deepseek_v41_flash_d_p.reference.packed_ref import forward_packed

    d = P.ChunkDescriptor.single(CHUNK, seg.slot, seg.start, seg.end)
    taps = {}
    forward_packed(ref, d, [ids], {seg.slot: st}, rec=lambda n, t: taps.__setitem__(n, t.clone()))
    return {b: t[: seg.length] for b, t in taps.items()}, st


def _maxdiff(a: dict, b: dict) -> tuple[float, str]:
    worst, where = 0.0, ""
    for n in sorted(set(a) & set(b)):
        x, y = a[n], b[n]
        if x.shape != y.shape:
            return float("inf"), f"{n} shape"
        d = float((x.double() - y.double()).abs().max()) if x.numel() else 0.0
        if d > worst or d != d:
            worst, where = d, n
    return worst, where


def _state_bad(s1, s2) -> list:
    bad = []
    for i in sorted(s1.window):
        for n in ("window", "ckv", "index_k", "partial", "topk", "cand", "tail"):
            a, b = getattr(s1, n).get(i), getattr(s2, n).get(i)
            if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                bad.append(f"L{i}.{n}")
    return bad


@pytest.mark.timeout(1800)
def test_reference_glued_vs_alone(rig):
    from models.demos.deepseek_v41_flash_d_p.reference.packed_ref import fixed_expert_tiles

    ref, sess, base, text = rig
    seg = lambda n: (n, P.Segment(sess[n][0], sess[n][1], sess[n][1] + len(sess[n][2])), sess[n][2])  # noqa: E731
    cp = lambda names: {n: copy.deepcopy(base[n]) for n in names}  # noqa: E731
    with fixed_expert_tiles():
        g1, s1 = _glued(ref, [seg(n) for n in GLUE1], cp(GLUE1))
        g1b, s1b = _glued(ref, [seg(n) for n in GLUE1], cp(GLUE1))
        for n in GLUE1:  # determinism
            d, w = _maxdiff(g1[n], g1b[n])
            assert d == 0.0 and not _state_bad(s1[n], s1b[n]), (n, w, d)
        g2, s2 = _glued(ref, [seg(n) for n in GLUE2], cp(GLUE2))
        for n in SESSIONS:
            alone, sa = _glued(ref, [seg(n)], cp([n]))
            for gx, sx in ((g1, s1), (g2, s2)):
                if n in gx:
                    d, w = _maxdiff(gx[n], alone[n])
                    assert d == 0.0, f"{n}: glued vs alone differs at {w} by {d}"
                    assert not _state_bad(sx[n], sa[n]), (n, _state_bad(sx[n], sa[n]))
            _, s, ids = seg(n)
            td, st = _today(ref, s, ids, copy.deepcopy(base[n]))
            d, w = _maxdiff(alone[n], td)
            assert d == 0.0, f"{n}: alone vs today differs at {w} by {d}"
            assert not _state_bad(sa[n], st), n
        # split: E (90 rows) split after 64 rows (block 64: its first part [128, 192), the rest [192, 218))
        _, sE, idsE = seg("E")
        first, cont = P.split(sE, 64 + 10, block=64)
        assert first.end == 192 and cont.start == 192, (first, cont)
        names = ["A", "B", "D"]
        stE = copy.deepcopy(base["E"])
        c1 = [seg(n) for n in names] + [("E", first, idsE[: first.length])]
        gs1, ss1 = _glued(ref, c1, {**cp(names), "E": stE})
        c2 = [("E", cont, idsE[first.length :]), seg("C")]
        gs2, ss2 = _glued(ref, c2, {"E": stE, **cp(["C"])})
        stA = copy.deepcopy(base["E"])
        a1, _ = _glued(ref, [("E", first, idsE[: first.length])], {"E": stA})
        a2, sa2 = _glued(ref, [("E", cont, idsE[first.length :])], {"E": stA})
        d1, w1 = _maxdiff(gs1["E"], a1["E"])
        d2, w2 = _maxdiff(gs2["E"], a2["E"])
        assert d1 == 0.0 and d2 == 0.0, (w1, d1, w2, d2)
        assert not _state_bad(ss2["E"], sa2["E"]), _state_bad(ss2["E"], sa2["E"])
        print("glued CPU rung: equivalence, alone == today, split == alone, determinism: all bit-identical")
