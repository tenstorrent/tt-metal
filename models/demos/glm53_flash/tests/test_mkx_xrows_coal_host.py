# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts' activation mover / weight streamer / compute (kernels/mkx/xa.cpp, xw.cpp, xc.cpp) and the stack's
per-kind galaxy defaults beside them, host only (no ttnn, no device). Five subjects:

MKX_XROWS_COAL (GLM53_MKX_XROWS_COAL=1, opt-in): xa reads the T decode rows of x as ONE span per face instead of one
32-B read per row per face.

* the byte model of both read schedules on bf16 tile pages (mkx_common.hpp's tile_elem face layout): for every
  T = 1..16 the coalesced reads leave CB_X byte-identical to the per-row reads (zero-filled tiles, garbage in the
  source's padding rows and faces 2 / 3), and CB_X untilizes to x's T rows over zeros -> every downstream byte (gate_up,
  SwiGLU, the m gather, the down, the all-reduce) is the knob-off one: bitwise, no reduction order changes;
* the issue counts (2 T KT_GU -> 2 KT_GU, KT_GU at T = 16) and the B = 4 / B = 16 probe fit they remove;
* the kernel source: the #else branch is the 44adcb9ba40 per-row loop verbatim, the coalesced branch's offsets / sizes;
* the host knob (mk_experts.xcoal, exercised from its source without importing ttnn): off by default, only with
  MKX_XROWS, added to xa's defines alone (xw / xc keep their define lists, hence their JIT hashes).

MKX_RAW_SYNC (GLM53_MKX_RAW_SYNC, default on; =0 = the knob-off binaries) and the multi-layer scratch race it closes.
In a stack program (MK_LAYERS > 1) xa of layer l works in place on its core's xw scratch after MKX_SEM_LIST: CB_RAW
(MKX_RW_XA: sel / rank / the weights tile) and the list page's weight rows (the scatter, the routing weight tiles);
neither page is flow-controlled against xa (xc pops the list right after reading n, CB_RAW is never pushed), and xw of
layer l + 1 writes into both before its router-tail hand-off.

* test_raw_sync_model[byte_map] (mkx_common.hpp's constants; the 8x4 XSL layout and the one-chip EP one): xw(l + 1)'s
  pre-hand-off writes into CB_RAW (the table row, the live page) miss every byte xa touches there; its list-page zero
  fill covers every weight row xa scatters / reads -> the colliding write;
* test_raw_sync_model[race_without_hand_back / hand_back_closes_all]: an exhaustive interleaving model of one reader
  core (weight streamer: dense phases + xw; compute: dense phases + xc; activation mover: xa) over 1-3 stacked layers:
  without the hand-back an expert-only reader (it sits out every dense phase) whose expert stream can end before xa's
  last access lets xw(l + 1) write while xa(l) still reads -- with MKX_RW_XA on AND off (off only shortens the window in
  time) -- while a reader gated by its dense phases, a long stream or a one-layer program cannot; with the hand-back no
  interleaving does and none deadlocks;
* test_kernel_source: where xw's wait and xa's reset sit; test_host_knob: raw_sync on by default, xw / xa defines only,
  MKX_RW_XA opt-in.

MKX_EARLY (GLM53_MKX_EARLY, a galaxy default; =0 = the knob-off binaries) with the folded shared expert (fold_y2,
MKX_Y2_INIT): y2 is read only by xc's add_y2 at the last group (n == 0: y2 is the output), so xc waits for it there, xa
reads it after the first two groups' routing-weight tiles, and (MKX_EARLY_XROWS) x's decode rows are issued before the
hand-off wait of the reader phase ahead of the experts.

* test_early_protocol: every interleaving of 2 readers' xa / xc over 2 stacked layers with mk_common.hpp's hand-off
  semantics (exact-count wait, then 0) and the CB depths of mk_experts.py: no violation and no deadlock with or without
  the knob; a late y2 read, an early y2 signal and an extra one are caught;
* test_early_kernel_source / test_early_xrows_source: the wait / read / hook positions, the guards, the shared row-read
  helper; test_early_x_is_final: the router phase waits for every hn2 producer before the hook's phase;
* test_early_host_knob / test_early_galaxy_default: the define (xa / xc only), its refusals, the galaxy default and an
  explicit 0.

GLM53_MHC_PRE_SPLIT (a galaxy default; stack_model.set_knobs puts it on the dense_kda / sparse_kda / sparse_dsa kinds):
the mhc_pre root sends pre before its 20-plane relayout. A per-kind knob that set_knobs pops before every kind, so the
launch value is read once per process.

* test_pre_split_galaxy_default: apply() on the 8x4, an explicit 0, other meshes;
* test_pre_split_set_knobs (a child with a stub ttnn: stack_model imports it): on for every kind at t 1 / 4 / 16 in any
  kind order, one "active" line per kind; an explicit 0 off everywhere; TP8 / one chip off; GLM53_STACK_KNOB_SET per
  kind without leaking; a later environment change ignored; a foreign value warned once and off. In the same child,
  GLM53_KDA_VERIFY_RSPLIT through set_knobs and the TP owner rewrites: a sparse_dsa PIPE / FLAT entry drops the
  FLAT / RSPLIT defaults before the KDA verify build reads them; every owner rewrite runs the lanes check on its own
  values;
* test_pre_split_knob_names: the stack's GLM53_MHC_PRE_SPLIT and the mesh model's GLM53_MESH_MHC_PRE_SPLIT never meet.

GLM53_MHC_RMS_EARLY (a galaxy default; the sparse_dsa programs' mhc_pre roots, kernels/mhc_pre/sync.cpp MHC_RMS_EARLY): the
compact coefficient tile, then round 2 (1/rms(h) to the readers), then the post / comb tiles. mhc_post reads coef without
a hand-off from the root, so the coefficient tile must land before the root's 1/rms(h) signal.

* test_rms_early_kernel_order: the coefficient tile's write + barrier, then the round-2 fragment (invh written, then the
  readers signalled), then the post / comb tiles; the knob-off branch keeps the old order (the fragment at the old
  place, the coefficient tile after post / comb); no MHC_RMS_EARLY with a cp build;
* test_rms_early_host_knob: the define on the root sync role only, a cp build refused, read by MkMla / MkMoE alone, one
  "active" line; test_rms_early_galaxy_default: apply() on the 8x4, an explicit 0, other meshes.

    pytest --noconftest models/demos/glm53_flash/tests/test_mkx_xrows_coal_host.py -q
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

TT = Path(__file__).resolve().parents[3] / "autoports" / "zai_org_glm_5_3_flash" / "tt"
XA = TT / "kernels" / "mkx" / "xa.cpp"
COMMON = TT / "kernels" / "mkx" / "mkx_common.hpp"
MKX = TT / "mk_experts.py"
XW = TT / "kernels" / "mkx" / "xw.cpp"
KT_GU = 128  # hidden 4096 / 32: x's K tiles (xa's KT_GU)
PAGE = 2048  # bf16 tile page
FACE = 512  # bf16 face (16 x 16 x 2 B)
TILE16 = PAGE  # a bf16 tile page in xw's routing scratch (the ids tile, the weights tile)
MEM_ZEROS_SIZE = 512  # Blackhole dev_mem_map.h: the NoC zero-fill granule
TOPK = 8
E = 288  # routed experts (MAX_EXPERTS)


def tile_elem(row: int, col: int) -> int:
    """mkx_common.hpp tile_elem: element (row, col) of a 32x32 tile in face layout, in elements"""
    return (((row >> 4) << 1) + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15)


def tilize(x: torch.Tensor) -> torch.Tensor:
    """[32, 32 KT] bf16 -> [KT, 2048] uint8 tile pages (face layout)"""
    kt = x.shape[1] // 32
    idx = torch.tensor([tile_elem(r, c) for r in range(32) for c in range(32)])
    pages = torch.empty(kt, 1024, dtype=torch.bfloat16)
    for k in range(kt):
        pages[k, idx] = x[:, 32 * k : 32 * (k + 1)].reshape(-1)
    return pages.view(torch.uint8).reshape(kt, PAGE)


def untilize(pages: torch.Tensor) -> torch.Tensor:
    kt = pages.shape[0]
    idx = torch.tensor([tile_elem(r, c) for r in range(32) for c in range(32)])
    el = pages.reshape(kt, PAGE).contiguous().view(torch.bfloat16).reshape(kt, 1024)
    return torch.cat([el[k, idx].reshape(32, 32) for k in range(kt)], dim=1)


def per_row_reads(t: int):
    """the knob-off schedule (xa.cpp #else): per tile, per row r < T, face 0 row r and face 1 row r -> (src, dst, bytes)"""
    return [(off, off, 32) for r in range(t) for off in (r * 32, FACE + r * 32)]


def coalesced_reads(t: int):
    """MKX_XROWS_COAL: one span per face; T = 16 = the tile's first 1 KB (faces 0 + 1)"""
    if t == 16:
        return [(0, 0, 2 * t * 32)]
    return [(0, 0, t * 32), (FACE, FACE, t * 32)]


def run_schedule(src: torch.Tensor, reads, t: int) -> torch.Tensor:
    """xa's CB_X after the zero fill (xrows_prologue / the standalone fill) and the schedule's NoC reads"""
    dst = torch.zeros_like(src)
    for k in range(src.shape[0]):
        for s, d, n in reads:
            assert s % 16 == 0 and d % 16 == 0 and n % 16 == 0, "NoC L1 reads: 16-B aligned"
            dst[k, d : d + n] = src[k, s : s + n]
    return dst


@pytest.mark.parametrize("t", list(range(1, 17)))
def test_coalesced_cb_x_bitwise(t):
    g = torch.Generator().manual_seed(1000 + t)
    x = torch.randn(32, 32 * KT_GU, generator=g).to(torch.bfloat16)
    # the source tile rows >= T are not x's: stale / garbage bytes (NaN / Inf patterns included); they must not land
    x[t:] = (
        torch.randint(0, 1 << 16, (32 - t, 32 * KT_GU), generator=g, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    src = tilize(x)
    a = run_schedule(src, per_row_reads(t), t)
    b = run_schedule(src, coalesced_reads(t), t)
    assert torch.equal(a, b), f"T={t}: CB_X bytes differ"
    want = torch.zeros(32, 32 * KT_GU, dtype=torch.bfloat16)
    want[:t] = x[:t]
    got = untilize(b)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), f"T={t}: CB_X is not x's rows over zeros"


@pytest.mark.parametrize("t", list(range(1, 17)))
def test_span_union_is_the_row_set(t):
    """the coalesced spans cover exactly the per-row byte set: rows < T of faces 0 / 1 by tile_elem, nothing else"""
    rows = {2 * tile_elem(r, c) + b for r in range(t) for c in range(32) for b in (0, 1)}
    old = {s + i for s, _, n in per_row_reads(t) for i in range(n)}
    new = {s + i for s, _, n in coalesced_reads(t) for i in range(n)}
    assert old == new == rows


def test_issue_counts_and_probe_fit():
    """reads per reader: 2 T KT_GU (knob off) vs 2 KT_GU (KT_GU at T = 16). The probe pair val_pk84_b4 / b16 (chip 0,
    layer 4, reader0_a xa marks 1 -> 2: 22.7 / 79.1 us; the same at layers 5 / 6 and on reader 2) fits a + b x reads
    with b = 18.4 ns per 32-B read (~25 cycles at 1.35 GHz: issue-bound) and a = 3.9 us (y2 seed, list, xinv)"""
    old = {t: KT_GU * len(per_row_reads(t)) for t in (1, 4, 16)}
    new = {t: KT_GU * len(coalesced_reads(t)) for t in (1, 4, 16)}
    assert old == {1: 256, 4: 1024, 16: 4096} and new == {1: 256, 4: 256, 16: 128}
    b = (79.1 - 22.7) / (old[16] - old[4])
    a = 22.7 - b * old[4]
    assert 0.0175 < b < 0.0195 and 3.0 < a < 4.5, (a, b)
    # knob-off issue time removed per MoE layer (the bytes themselves: 128 KB per reader at T = 16, ~3 us at 32 B/clk)
    assert round(b * (old[16] - new[16])) == 73 and round(b * (old[4] - new[4])) == 14


# ---- MKX_RAW_SYNC: the byte map and the interleaving model ----------------------------------------------------
def _consts() -> dict:
    src = COMMON.read_text()
    c = {k: int(v) for k, v in re.findall(r"constexpr uint32_t (\w+) = (\d+);", src)}
    assert "constexpr uint32_t LIVE_OFF = RAW_BYTES - 128;" in src
    assert "constexpr uint32_t LIST_XA_WORD = LIST_BYTES / 4 - 4;" in src
    c["LIVE_OFF"] = c["RAW_BYTES"] - 128
    c["LIST_XA_WORD"] = c["LIST_BYTES"] // 4 - 4
    return c


def _disjoint(a, b) -> bool:
    return a[1] <= b[0] or b[1] <= a[0]


def _byte_map():
    """xw.cpp's routing-step layout (MKX_RW: N_SC = 1, no MKX_LOC) for the 8x4 XSL chip (E_LOC 72) and the one-chip
    EP program (E_LOC 288), every native t"""
    c = _consts()
    assert c["MAX_EXPERTS"] == E and c["RW_TOPK_XA"] == TOPK
    src = XW.read_text()
    assert "const uint32_t tab_l1 = raw + (1 + N_SC) * TILE16;" in src
    assert "const uint32_t work = tab_l1 + ((2 * E * 4 + 63) & ~63u);" in src
    assert "const uint32_t rows = T * TOPK < E_LOC ? T * TOPK : E_LOC;" in src
    for e_loc in (72, 288):
        tab = (1 + 1) * TILE16
        work = tab + ((2 * E * 4 + 63) & ~63)
        for t in (1, 4, 8, 12, 14, 16, 32):
            # ---- CB_RAW: the pre-hand-off writes vs every byte xa reads (sel [b < t], rank [E], the weights tile)
            pre = [(tab, tab + 2 * e_loc * 4), (c["LIVE_OFF"], c["RAW_BYTES"])]  # the table row, the live page
            xa_raw = [
                (TILE16, 2 * TILE16),
                (work, work + 4 * TOPK * t),
                (work + 32 * TOPK * 4, work + 32 * TOPK * 4 + 2 * E),
            ]
            assert all(_disjoint(p, r) for p in pre for r in xa_raw), (e_loc, t)
            # ---- the list page: the weight-row zero fill vs xa's weight rows (rank / group slot < n <= min(E_LOC, 8 t))
            wl_off = (c["LIST_IDS"] + c["MAX_EXPERTS"]) * 4
            z0 = wl_off & ~15
            rows = min(t * TOPK, e_loc)
            fill = (z0, z0 + ((wl_off - z0 + rows * 64 + MEM_ZEROS_SIZE - 1) & ~(MEM_ZEROS_SIZE - 1)))
            weights = (wl_off, wl_off + rows * 64)  # wl[rank * 32 + b], b < 32 (the EP CB_RS loop reads all 32 rows)
            assert fill[0] <= weights[0] and weights[1] <= fill[1] <= c["LIST_BYTES"], (e_loc, t, fill, weights)
            # n / T (words 0, 1) and xa's scratch pointers (LIST_XA_WORD ..) are written after the hand-off only
            assert _disjoint(fill, (0, 8)) and _disjoint(fill, (4 * c["LIST_XA_WORD"], 4 * c["LIST_XA_WORD"] + 8))


# ---- the interleaving model --------------------------------------------------------------------------------------
# shared state: CB_LIST pages, MKX_SEM_LIST, the expert ring, a dense ring, CB_RS, CB_MO, CB_M, the xa-done bits (the
# owners' P10(l) waits for every reader's xa(l): the hand-offs of layer l + 1 follow it), the layer of the last xw write
# per region (the weight rows; sel / rank / weights tile and n / T are written after the hand-off only), a violation
LIST, FLAG, RING, DRING, RS, MO, M, DONE, G_WL, G_POST, VIOL = range(11)
RD = 2  # dense ring slots


def _streams(groups):
    """the stream / compute order: gate_up(0), then gate_up(p) before down(p - 1), down(last) last"""
    order = [("gu", 0)]
    for p in range(1, len(groups)):
        order += [("gu", p), ("dn", p - 1)]
    return order + [("dn", len(groups) - 1)]


def _procs(*, layers: int, sync: bool, rw_xa: bool, dense: int, ring: int, groups: list):
    """groups: per expert group (pos) of this reader (G, D): G gate_up blocks before its routing weight tile (None = past
    the union: a zero m tile, no blocks, no weight tile), D down blocks"""
    stack, w, c, a = layers > 1, [], [], []
    for lay in range(layers):
        # weight streamer: the dense phases (never waits for activations, only ring space), then xw
        w += [("push", DRING, RD)] * dense
        if sync and stack:
            w += [("eq", FLAG, 0)]  # MKX_RAW_SYNC: xa(l - 1) handed the scratch back
        w += [("eq", LIST, 0), ("write", G_WL, lay)]  # cb_reserve_back(CB_LIST), the weight-row zero fill
        w += [("done", lay - 1), ("write", G_POST, lay), ("write", G_WL, lay)]  # router tail(l): ids / union / list
        w += [("push", LIST, 1), ("set", FLAG, 1)]
        for kind, p in _streams(groups):
            g, d = groups[p]
            w += [("push", RING, ring)] * ((g or 0) if kind == "gu" else d)
        # compute: the dense phases (on this layer's activations), then xc
        if dense:
            c += [("done", lay - 1)] + [("pop", DRING)] * dense
        c += [("pop", LIST)]  # n read, the list page popped
        for kind, p in _streams(groups):
            g, d = groups[p]
            if kind == "gu":
                c += [("push", MO, None)] if g is None else [("pop", RING)] * g + [("pop", RS), ("push", MO, None)]
            else:
                c += [("pop", M)] + [("pop", RING)] * d
        # activation mover: xa
        a += [("done", lay - 1), ("eq", FLAG, 1)]
        if stack and not sync:
            a += [("set", FLAG, 0)]  # the reset right after the wait (knob off)
        a += [("read", G_POST, lay)]  # n, T
        if rw_xa:  # the scatter: sel / rank / the weights tile -> the weight rows (two steps: a point inside it)
            a += [("read", G_POST, lay), ("read", G_WL, lay), ("read", G_POST, lay), ("read", G_WL, lay)]

        def rs(p):
            return [] if groups[p][0] is None else [("read", G_WL, lay), ("push", RS, None)]

        a += rs(0)
        for p in range(len(groups)):
            a += rs(p + 1) if p + 1 < len(groups) else []
            a += [("pop", MO), ("push", M, 1)]
        if stack and sync:
            a += [("set", FLAG, 0)]  # MKX_RAW_SYNC: the hand-back after the last access
        a += [("signal", lay)]
    return [w, c, a]


def _enabled(op, s) -> bool:
    k = op[0]
    if k == "eq":
        return s[op[1]] == op[2]
    if k == "push":
        return op[2] is None or s[op[1]] < op[2]
    if k == "pop":
        return s[op[1]] >= 1
    if k == "done":
        return op[1] < 0 or bool(s[DONE] >> op[1] & 1)
    return True


def _apply(op, s) -> list:
    s = list(s)
    k = op[0]
    if k == "push":
        s[op[1]] += 1
    elif k == "pop":
        s[op[1]] -= 1
    elif k == "set":
        s[op[1]] = op[2]
    elif k == "write":
        s[op[1]] = op[2]
    elif k == "read":  # xa(l) reads / writes a region: it must still hold layer l's routing step
        if s[op[1]] != op[2]:
            s[VIOL] = 1
    elif k == "signal":
        s[DONE] |= 1 << op[1]
    return s


def _explore(**kw) -> dict:
    """every interleaving of the three RISCs (DFS over the reachable states): violations (an xa access after the next
    layer's xw wrote that region), deadlocks (no RISC can move, not all finished), finished states"""
    procs = _procs(**kw)
    init = (0, 0, 0) + (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0)
    seen, todo = {init}, [init]
    out = dict(viol=0, dead=0, done=0, states=0)
    while todo:
        st = todo.pop()
        pcs, sh = st[:3], st[3:]
        if sh[VIOL]:
            out["viol"] += 1
            continue
        moved = False
        for i in range(3):
            if pcs[i] < len(procs[i]) and _enabled(procs[i][pcs[i]], sh):
                moved = True
                nxt = tuple(pcs[j] + (j == i) for j in range(3)) + tuple(_apply(procs[i][pcs[i]], sh))
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        if all(pcs[i] == len(procs[i]) for i in range(3)):
            out["done"] += 1
        elif not moved:
            out["dead"] += 1
    out["states"] = len(seen)
    return out


SHORT = [(2, 1)]  # one group, the reader's shard inside the union: stream 3 blocks <= ring 2 + the 2 gate_up blocks
PAST = [(None, 1)]  # one group, the reader's shards past the union (q >= n): only down blocks, no weight tile read
LONG = [(2, 3)]  # one group, 5 blocks: the stream ends only after a down block, i.e. after xa's gather
TWO = [(2, 1), (2, 1)]  # two groups: the second weight tile is read after the second gate_up's blocks drained


def _race_without_hand_back():
    """knob off (903900a212f's protocol): the race is reachable exactly where the device saw it"""
    base = dict(sync=False, ring=2)
    for rw_xa in (True, False):
        # an expert-only reader (dense 0) whose stream can end before xa's last weight-tile read: reachable with the
        # scatter on xa AND on xw (MKX_RW_XA off narrows the window in time, it does not close it)
        for g in (SHORT, TWO):
            assert _explore(layers=2, rw_xa=rw_xa, dense=0, groups=g, **base)["viol"] > 0, (rw_xa, g)
        # ... not with a stream that ends only after a down block (= after xa's gather, its last access)
        r = _explore(layers=2, rw_xa=rw_xa, dense=0, groups=LONG, **base)
        assert r["viol"] == 0 and r["dead"] == 0 and r["done"] > 0, (rw_xa, r)
        # a reader gated by its dense phases (more blocks than its ring: the next layer's compute needs P10(l))
        for g in (SHORT, PAST, LONG, TWO):
            r = _explore(layers=2, rw_xa=rw_xa, dense=RD + 1, groups=g, **base)
            assert r["viol"] == 0 and r["dead"] == 0, (rw_xa, g, r)
        # one sparse layer per program: no next-layer xw
        for g in (SHORT, PAST):
            assert _explore(layers=1, rw_xa=rw_xa, dense=0, groups=g, **base)["viol"] == 0
    # shards past the union read no weight tile: only the scatter (MKX_RW_XA) is exposed there
    assert _explore(layers=2, rw_xa=True, dense=0, groups=PAST, **base)["viol"] > 0
    assert _explore(layers=2, rw_xa=False, dense=0, groups=PAST, **base)["viol"] == 0


def _hand_back_closes_all():
    """MKX_RAW_SYNC: no violation, no deadlock, every run finishes -- 2 and 3 layers, RW_XA on / off, expert-only and
    gated readers, every stream shape, rings of 1 .. 3 slots"""
    for layers in (2, 3):
        for rw_xa in (True, False):
            for dense in (0, RD + 1):
                for g in (SHORT, PAST, LONG, TWO):
                    for ring in (1, 2, 3):
                        r = _explore(layers=layers, sync=True, rw_xa=rw_xa, dense=dense, ring=ring, groups=g)
                        assert r["viol"] == 0 and r["dead"] == 0 and r["done"] > 0, (layers, rw_xa, dense, g, ring, r)


@pytest.mark.parametrize("case", ["byte_map", "race_without_hand_back", "hand_back_closes_all"])
def test_raw_sync_model(case):
    cases = {"byte_map": _byte_map, "race_without_hand_back": _race_without_hand_back}
    cases["hand_back_closes_all"] = _hand_back_closes_all
    cases[case]()


def _branches(src: str):
    m = re.search(r"#if defined\(MKX_XROWS_COAL\)\n(.*?)#else\n(.*?)#endif\n", src, re.S)
    assert m, "xa.cpp: the MKX_XROWS_COAL #if / #else / #endif block"
    return m.group(1), m.group(2)


OLD_LOOP = """        for (uint32_t k = 0; k < KT_GU; ++k) {
            const uint32_t dst = xl1 + k * xpage;
            for (uint32_t r = 0; r < MKX_XROWS; ++r) {
                noc_async_read(x.get_noc_addr(k, r * 32), dst + r * 32, 32);
                noc_async_read(x.get_noc_addr(k, 512 + r * 32), dst + 512 + r * 32, 32);
            }
        }
"""


SYNC_IF = "#if defined(MKX_RAW_SYNC) && defined(MK_LAYERS) && MK_LAYERS > 1\n"


def _block(src: str, start: int) -> str:
    return src[start : src.index("#endif", start)]


def test_kernel_source():
    src = XA.read_text()
    # the row reads: xrows_read (one helper for the expert phase and xrows_early); its #else branch is the 44adcb9ba40
    # per-row loop verbatim, one indent level up
    h = src.index("FORCE_INLINE void xrows_read(const XAcc& x, uint32_t xl1, uint32_t xpage) {")
    coal, off = _branches(src[h:])
    assert "\n".join(("    " + x) if x else x for x in off.split("\n")) == OLD_LOOP, "the knob-off per-row loop"
    assert src.count("#if defined(MKX_XROWS_COAL)") == 1 and h < src.index("#if defined(MKX_XROWS_COAL)")
    # the expert phase calls it inside MKX_XROWS, between the zero-fill barrier and the push's barrier
    k = src.index("void kernel_main() {")
    i = src.index("        xrows_read<KT_GU>(x, xl1, xpage);\n", k)
    assert src.index("#if defined(MKX_XROWS)\n    // decode rows only") < i < src.index("#elif defined(MKX_XPROG)")
    assert src[:i].endswith("noc_async_read_barrier();  // the zeros land before the row reads into the same tiles\n")
    rest = src[i:].split("\n", 1)[1].removeprefix("#endif  // MKX_EARLY_XROWS\n").lstrip()
    assert rest.startswith("noc_async_read_barrier();\n        cb_push_back(CB_X, KT_GU);")
    assert "constexpr uint32_t XROW_BYTES = MKX_XROWS * 32;" in coal
    reads = re.findall(r"noc_async_read\(x\.get_noc_addr\(k, (\d+)\), (dst(?: \+ \d+)?), ([^)]+)\);", coal)
    assert reads == [("0", "dst", "2 * XROW_BYTES"), ("0", "dst", "XROW_BYTES"), ("512", "dst + 512", "XROW_BYTES")]
    assert "if constexpr (MKX_XROWS == 16)" in coal
    # the face layout the byte model uses is the kernels' one
    assert "return (((row >> 4) << 1) + (col >> 4)) * 256 + (row & 15u) * 16 + (col & 15u);" in COMMON.read_text()
    # MKX_RAW_SYNC
    xw, xa = XW.read_text(), XA.read_text()
    # xw: the wait precedes its first write into CB_RAW / the list page (both before the hand-off wait)
    assert xw.count(SYNC_IF) == 1
    i = xw.index(SYNC_IF)
    blk = _block(xw, i)
    assert "get_semaphore(MKX_SEM_LIST)" in blk and "while (*xa_flag != 0) {" in blk and "invalidate_l1_cache();" in blk
    order = [
        "shard[s] = get_arg_val<uint32_t>(ph_rt0 + 5 + s);",
        SYNC_IF,
        "cb_reserve_back(CB_RAW, 1);",
        "noc_async_read_page(0, tab_t, tab_l1);",
        "cb_reserve_back(CB_LIST, 1);",
        "noc_async_read(zeros, list_l1 + Z0 + off, MEM_ZEROS_SIZE);",
        "mk::wait_handoff(MKX_WAIT_SEM",
        "*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MKX_SEM_LIST)) = 1;",
    ]
    pos = [xw.index(s) for s in order]
    assert pos == sorted(pos), order
    # xa: knob off keeps the reset right after the wait; knob on resets after the last access, outside the gather loop
    early = "#if defined(MK_LAYERS) && MK_LAYERS > 1 && !defined(MKX_RAW_SYNC)\n    *flag = 0;"
    assert xa.count(early) == 1 and xa.index("while (*flag == 0) {") < xa.index(early)
    assert xa.count(SYNC_IF) == 1
    j = xa.index(SYNC_IF)
    assert "*flag = 0;" in _block(xa, j) and 'asm volatile("" ::: "memory");' in _block(xa, j)
    assert xa[:j].endswith("        mk::signal_handoff(sem_mr, ph_rt0 + COORDS, NREAD);\n    }\n"), "after the pos loop"
    for tok in ("wl[", "sel[", "rank[", "sc16[", "list[", "rs32[", "rs["):
        assert xa.rindex(tok) < j, tok
    assert j < xa.index("// the output: 2 tiles per owned shard")
    assert xa.count("*flag = 0;") == 2  # the two exclusive resets, nothing else writes the flag


def _host_fn(name: str):
    """a mk_experts.py function exercised from its source (importing the module would import ttnn)"""
    tree = ast.parse(MKX.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MKX), "exec"), ns)  # noqa: S102 (our own source)
    return ns[name]


def test_host_knob(monkeypatch):
    xcoal = _host_fn("xcoal")
    monkeypatch.delenv("GLM53_MKX_XROWS_COAL", raising=False)
    assert not xcoal([("MKX_XROWS", "16")]), "default off"
    monkeypatch.setenv("GLM53_MKX_XROWS_COAL", "0")
    assert not xcoal([("MKX_XROWS", "16")])
    monkeypatch.setenv("GLM53_MKX_XROWS_COAL", "1")
    assert xcoal([("MKX_XROWS", "16")]) and xcoal([("MKX_XROWS", "1")])
    assert not xcoal([]) and not xcoal([("MKX_XPROG", "1")]), "no MKX_XROWS (T > 16 / GLM53_MKX_XROWS=0): no define"
    src = MKX.read_text()
    split = src.index("wdefs, adefs = list(defs), list(defs)")
    guard = src.index("if xcoal(xprog):")
    add = src.index('adefs.append(("MKX_XROWS_COAL", "1"))')
    assert split < guard < add < src.index("wtail, atail = [], []"), "xa's defines alone, after the w / a split"
    assert src.count('MKX_XROWS_COAL", "1"') == 1 and "cdefs, a_ct_y2, y2_rt = list(defs)" in src
    # no revision bump: a knob-off build keeps MKX_REV (every expert kernel's define list unchanged)
    assert re.search(r"^MKX_REV = 24 ", src, re.M)
    # MKX_RAW_SYNC: on by default (device chain_mk184), =0 = the knob-off binaries
    raw_sync = _host_fn("raw_sync")
    monkeypatch.delenv("GLM53_MKX_RAW_SYNC", raising=False)
    assert raw_sync(), "default on"
    monkeypatch.setenv("GLM53_MKX_RAW_SYNC", "0")
    assert not raw_sync()
    monkeypatch.setenv("GLM53_MKX_RAW_SYNC", "1")
    assert raw_sync()
    guard = src.index("if raw_sync():")
    assert split < guard < src.index("wtail, atail = [], []"), "xw / xa defines only, after the w / a split"
    assert src[guard:].startswith(
        'if raw_sync():\n            wdefs.append(("MKX_RAW_SYNC", "1"))\n            adefs.append(("MKX_RAW_SYNC", "1"))\n'
    )
    assert src.count('"MKX_RAW_SYNC", "1"') == 2
    # MKX_RW_XA is opt-in since 2026-11-10 (the race of test_raw_sync_model); MKX_REV unchanged (above)
    assert 'os.environ.get("GLM53_MKX_RW_XA", "0") == "1"' in src


# ---- GLM53_MKX_EARLY (xa.cpp / xc.cpp MKX_EARLY, mk_experts.early_defines; a galaxy default) ---------------------------
# The rule: y2 (the folded shared expert's down output, fold_y2 / MKX_Y2_INIT) is read only by xc's add_y2 at the last
# group, or copied to the output when no local expert is selected (n == 0). With the knob xc waits for CB_Y2 there, xa
# reads y2 after the first two groups' routing-weight tiles, and (MKX_EARLY_XROWS, with the coalesced row reads in a
# megakernel composite) x's decode rows are issued before the hand-off wait of the reader phase ahead of the experts.

XC = TT / "kernels" / "mkx" / "xc.cpp"
RA = TT / "kernels" / "mk" / "reader_a.cpp"
RMA = TT / "kernels" / "mk" / "reader_moe_a.cpp"
GD_SRC = TT / "galaxy_defaults.py"
Y2_WAIT = "cb_wait_front(CB_Y2, 2 * nsh);"

# the y2 hand-off model: R readers' activation mover (xa) and compute (xc) over stacked layers, every interleaving (a DFS
# over the reachable states, as _explore). Its operations are the kernels' (mk_common.hpp): signal_handoff = +1 on every
# reader's semaphore; wait_handoff = wait until the semaphore is EXACTLY the count, then set it to 0; CB pushes / pops /
# waits in pages against the depths parts_raw gives the CBs (mk_experts.py, read from its source). The layers are
# separated by the all-reduce: no reader starts layer l + 1 (its shared down rewrites y2) before every reader's xa(l)
# wrote its output. The y2 a reader reads must hold layer l on every reader (each reader's y2 tiles are written by the
# other readers' shared down). Per reader: CBs X, RS, MO, M, Y2C, Y; semaphores Y2S, MW0, MW1, MR0, MR1; SD = the layer
# of its last shared-down write, DN = the layer of its last output; then one violation flag.
_CBS, _SEMS = ("X", "RS", "MO", "M", "Y2C", "Y"), ("Y2S", "MW0", "MW1", "MR0", "MR1")
_CB_INDEX = dict(M=2, MO=4, RS=5, Y=8, Y2C=11)  # mkx_common.hpp: CB_M / CB_MO / CB_RS / CB_Y / CB_Y2 = MKX_CB0 + i


def _cb_depths(nsh: int, m_buffers: int) -> dict:
    """the page counts parts_raw gives CB_M / CB_MO / CB_RS / CB_Y / CB_Y2 (mk_experts.py's cb(...) list, evaluated
    for the batched expert TP with nsh owned shards); m_tiles = I tiles = 64"""
    import types

    src = MKX.read_text()
    consts = dict(re.findall(r"^(XSL_RS_PAGES) = (\d+)", src, re.M))
    me = types.SimpleNamespace(m_buffers=m_buffers, m_tiles=64, xsl=0)
    ns = {"nsh": nsh, "self": me, "XSL_RS_PAGES": int(consts["XSL_RS_PAGES"])}
    out = {}
    for name, i in _CB_INDEX.items():
        exprs = re.findall(rf"\bcb\({i}, [^,]+, [^,]+, ([^)]+)\)", src)
        assert len(exprs) == 1, (name, exprs)
        out[name] = eval(exprs[0], {}, ns)  # noqa: S307 (our own source)
    return out


def _y2_slots(R: int) -> dict:
    names = list(_CBS) + list(_SEMS) + ["SD", "DN"]
    return {(nm, r): i * R + r for i, nm in enumerate(names) for r in range(R)} | {"VIOL": len(names) * R}


def _y2_procs(*, readers: list, n: int, layers: int, early: bool, mbuf: int, mut: str = "") -> list:
    """[xa(0), xc(0), xa(1), xc(1), ...]; readers = each reader's owned shards (shard j carries the group's expert
    j // 8, MKX_XSL_W 8); n = the chip's union. mut: "late" = y2 read after xa waited for the output; "early_sig" = the
    y2 signal before the shared-down write; "extra_sig" = reader 0 signals twice; "xc_nowait" = xc adds y2 without
    waiting for it"""
    R, S = len(readers), _y2_slots(len(readers))
    npos = (n + 7) // 8
    out = []
    for r, shards in enumerate(readers):
        nsh = len(shards)
        dep = _cb_depths(nsh, mbuf)
        has = lambda p, sh: 8 * p + sh // 8 < n  # noqa: E731
        c = lambda nm: S[(nm, r)]  # noqa: E731
        xa, xc = [], []
        for lay in range(layers):
            read_y2 = [("wexact", c("Y2S"), R), ("y2chk", lay), ("push", c("Y2C"), 2 * nsh, dep["Y2C"])]
            xa += [("bar", lay - 1)]
            sig = [("sig", "Y2S")] * (2 if (mut == "extra_sig" and r == 0) else 1)
            sd = [("set", c("SD"), lay)]
            xa += sig + sd if mut == "early_sig" else sd + sig  # signal_handoff(MKX_Y2_SEM) at the expert phase start
            xa += [("push", c("X"), 1, 1)]
            if not early:
                xa += read_y2  # knob off: before the list
            rs = lambda p: [("push", c("RS"), 1, dep["RS"]) for sh in shards if has(p, sh)]  # noqa: E731
            xa += rs(0) if npos else []
            if early and npos == 0 and mut != "late":
                xa += read_y2
            for pos in range(npos):
                p = pos & 1
                xa += rs(pos + 1) if pos + 1 < npos else []
                if early and pos == 0 and mut != "late":
                    xa += read_y2
                xa += [("wexact", c(f"MR{p}"), R)] if pos >= 2 else []
                xa += [("pop", c("MO"), 1)] * nsh + [("sig", f"MW{p}"), ("wexact", c(f"MW{p}"), R)]
                xa += [("push", c("M"), 64, dep["M"]), ("sig", f"MR{p}")]
            for pos in range(max(npos - 2, 0), npos):  # stack: the last two positions' read acknowledgements
                xa += [("wexact", c(f"MR{pos & 1}"), R)]
            xa += [("pop", c("Y"), 2 * nsh)]
            xa += read_y2 if (early and mut == "late") else []
            xa += [("set", c("DN"), lay)]
            # compute
            xc += [("wait", c("X"), 1)]
            if (not early or npos == 0) and mut != "xc_nowait":
                xc += [("wait", c("Y2C"), 2 * nsh)]
            xc += ([("need", c("Y2C"), 2 * nsh)] + [("push", c("Y"), 2, dep["Y"])] * nsh) if npos == 0 else []

            def gu(p):
                ops = []
                for sh in shards:  # per owned shard: SwiGLU x its weight tile (an expert of the union), then m
                    ops += ([("pop", c("RS"), 1)] if has(p, sh) else []) + [("push", c("MO"), 1, dep["MO"])]
                return ops

            def dn(p):
                last = p + 1 == npos
                w = [("wait", c("M"), 64)] + (
                    [("wait", c("Y2C"), 2 * nsh)] if (early and last and mut != "xc_nowait") else []
                )
                y = [("need", c("Y2C"), 2 * nsh)] + [("push", c("Y"), 2, dep["Y"])] * nsh  # add_y2, then the output
                return w + (y if last else []) + [("pop", c("M"), 64)]

            if npos:
                xc += gu(0)
                for pos in range(1, npos):
                    xc += gu(pos) + dn(pos - 1)
                xc += dn(npos - 1)
            xc += [("pop", c("Y2C"), 2 * nsh), ("pop", c("X"), 1)]
        out += [xa, xc]
    return out


def _y2_enabled(op, s, R, S) -> bool:
    k = op[0]
    if k == "push":
        return s[op[1]] + op[2] <= op[3]
    if k in ("pop", "wait"):
        return s[op[1]] >= op[2]
    if k == "wexact":
        return s[op[1]] == op[2]
    if k == "bar":
        return op[1] < 0 or all(s[S[("DN", q)]] >= op[1] for q in range(R))
    return True


def _y2_apply(op, s, R, S) -> tuple:
    s = list(s)
    k = op[0]
    if k == "push":
        s[op[1]] += op[2]
    elif k == "pop":
        s[op[1]] -= op[2]
    elif k == "wexact":
        s[op[1]] = 0
    elif k == "set":
        s[op[1]] = op[2]
    elif k == "sig":
        for q in range(R):
            s[S[(op[1], q)]] += 1
    elif k == "y2chk" and any(s[S[("SD", q)]] != op[1] for q in range(R)):
        s[S["VIOL"]] = 1  # y2 read before a reader's shared down wrote this layer's tiles (or after the next layer's)
    elif k == "need" and s[op[1]] < op[2]:
        s[S["VIOL"]] = 1  # add_y2 / the n == 0 copy on a CB_Y2 that has not landed
    return tuple(s)


def _y2_explore(*, readers: list, **kw) -> dict:
    """every interleaving: violations (y2 read before a reader's shared down wrote this layer's, or after the next
    layer's), deadlocks, finished states"""
    R, S = len(readers), _y2_slots(len(readers))
    procs = _y2_procs(readers=readers, **kw)
    init = [0] * (S["VIOL"] + 1)
    for q in range(R):
        init[S[("SD", q)]] = init[S[("DN", q)]] = -1
    init = tuple([0] * len(procs)) + tuple(init)
    seen, todo, n = {init}, [init], len(procs)
    out = dict(viol=0, dead=0, done=0)
    while todo:
        st = todo.pop()
        pcs, sh = st[:n], st[n:]
        if sh[S["VIOL"]]:
            out["viol"] += 1
            continue
        moved = False
        for i in range(n):
            if pcs[i] < len(procs[i]) and _y2_enabled(procs[i][pcs[i]], sh, R, S):
                moved = True
                nxt = tuple(pcs[j] + (j == i) for j in range(n)) + _y2_apply(procs[i][pcs[i]], sh, R, S)
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        if all(pcs[i] == len(procs[i]) for i in range(n)):
            out["done"] += 1
        elif not moved:
            out["dead"] += 1
    return out


# a 3-shard reader (bank 0, index 0: shards 0 / 24 / 48 -> experts 0 / 3 / 6 of a group) and a 2-shard one (bank 2,
# index 2: shards 18 / 42 -> experts 2 / 5); unions with no group, a partial group, a full one, 2 and 3 groups
Y2_READERS = [[0, 24, 48], [18, 42]]
Y2_UNIONS = (0, 1, 3, 8, 9, 20)


@pytest.mark.parametrize("early", [False, True])
def test_early_protocol(early):
    """2 readers x 2 stacked layers, the unions above, CB_M 1 or 2 gathers deep (GLM53_MKX_MBUF; the sparse stack's
    default 1): no violation and no deadlock with or without MKX_EARLY; the model catches a y2 read after xa's output
    wait (a deadlock), an early y2 signal (a read before that reader's shared down landed) and an extra signal (the
    exact-count wait overshoots); a compute that copies y2 out (n == 0) without its wait reads an unlanded CB_Y2"""
    common = COMMON.read_text()
    for nm, i in _CB_INDEX.items():
        assert f"constexpr uint32_t CB_{nm.replace('Y2C', 'Y2')} = MKX_CB0 + {i};" in common, nm
    assert _cb_depths(3, 1) == dict(M=64, MO=6, RS=6, Y=6, Y2C=6) and _cb_depths(2, 2)["M"] == 128
    assert re.search(r'm_buffers=int\(os\.environ\.get\("GLM53_MKX_MBUF", "1"\)\)', (TT / "mk_sparse.py").read_text())
    for n in Y2_UNIONS:
        for mbuf in (1, 2):
            r = _y2_explore(readers=Y2_READERS, n=n, layers=2, early=early, mbuf=mbuf)
            assert r["viol"] == 0 and r["dead"] == 0 and r["done"] > 0, (early, n, mbuf, r)
    for n in (0, 3, 9):
        if early:
            assert _y2_explore(readers=Y2_READERS, n=n, layers=2, early=True, mbuf=1, mut="late")["dead"] > 0, n
        r = _y2_explore(readers=Y2_READERS, n=n, layers=2, early=early, mbuf=1, mut="early_sig")
        assert r["viol"] > 0, (n, r)
        r = _y2_explore(readers=Y2_READERS, n=n, layers=2, early=early, mbuf=1, mut="extra_sig")
        assert r["viol"] > 0 or r["dead"] > 0, (n, r)
    # n == 0: the compute copies y2 out at once, so a missing wait reads an unlanded CB_Y2; with n > 0 xa's y2 read also
    # precedes its first CB_M push, so the last group's wait is implied by the gather order as well
    assert _y2_explore(readers=Y2_READERS, n=0, layers=2, early=early, mbuf=1, mut="xc_nowait")["viol"] > 0


def test_early_kernel_source():
    xa, xc = XA.read_text(), XC.read_text()
    common = (TT / "kernels" / "mk" / "mk_common.hpp").read_text()
    # the model's semantics are mk_common.hpp's: an exact-count wait, then 0
    wh = common[common.index("FORCE_INLINE void wait_handoff(") :]
    assert "noc_semaphore_wait(s, count);\n    noc_semaphore_set(s, 0);" in wh[: wh.index("\n}\n")]
    # xc: knob off waits at the start; knob on at n == 0 and at the XSL down's last group, before both XSL add_y2s
    assert xc.count(Y2_WAIT) == 3
    off = xc.index("#if !defined(MKX_EARLY)\n    " + Y2_WAIT + "\n#endif\n    if (n == 0) {")
    on0 = xc.index("if (n == 0) {\n#if defined(MKX_EARLY)\n        " + Y2_WAIT)
    assert off < on0 < xc.index("auto gate_up = [&]()")
    last = xc.index("#if defined(MKX_EARLY)\n            if (last) {\n                " + Y2_WAIT)
    xsl = xc.index("const uint32_t nq = n - GS * pos < GS ? n - GS * pos : GS;")
    assert xsl < last < xc.index("#if defined(MKX_XSL_SEED)", xsl)
    end = xc.index("        cb_wait_front(CB_RS, 1);\n        const bool first = pos == 0, last = pos + 1 == n;", xsl)
    adds = [xsl + m.start() for m in re.finditer(r"if \(Y2 && last\) \{\n\s+add_y2\(s\);", xc[xsl:end])]
    assert len(adds) == 2 and all(last < a for a in adds), adds
    assert "cb_pop_front(CB_Y2, 2 * nsh);" in xc[xc.index("down(npos - 1);") :]
    guard = '#error "MKX_EARLY: the batched expert TP (MKX_XSL) with the folded shared expert (MKX_Y2_INIT) only"'
    assert xc.count(guard) == 1 and xa.count(guard) == 1 and xc.index(guard) < xc.index("void kernel_main() {")
    # xa: the knob-off y2 block where it was (after x, before the list); knob on: read_y2() after push_rs(0) when
    # npos == 0, else in the gather loop at pos 0 after push_rs(1), before the first gather
    assert "#if defined(MKX_Y2_INIT) && !defined(MKX_EARLY)\n    mk::wait_handoff(MKX_Y2_SEM, NREAD);" in xa
    blk = xa[xa.index("auto read_y2 = [&]() {") :]
    assert blk.index("mk::wait_handoff(MKX_Y2_SEM, NREAD);") < blk.index("cb_push_back(CB_Y2, 2 * nsh);")
    sig = xa.index("mk::signal_handoff(MKX_Y2_SEM, ph_rt0 + COORDS, NREAD);")
    assert sig < xa.index("cb_reserve_back(CB_X, KT_GU);")  # (one Y2 signal, at the expert phase start)
    assert xa.count("signal_handoff(MKX_Y2_SEM") == 1
    assert xa.count("read_y2();") == 2 and xa.count("auto read_y2 = [&]()") == 1
    calls = [m.start() for m in re.finditer(r"read_y2\(\);", xa)]
    assert xa.index("        push_rs(0);\n    }\n#if defined(MKX_EARLY)\n    if (npos == 0) {") < calls[0]
    assert (
        xa.index("            push_rs(pos + 1);\n        }\n#if defined(MKX_EARLY)\n        if (pos == 0) {") < calls[1]
    )
    assert calls[1] < xa.index("// m all-gather through buffer p") < xa.index("cb_wait_front(CB_Y, 2 * nsh);")
    # every MKX_EARLY line is a guard, a comment, the #error or the macro defined under the guard: knob off preprocesses
    # to the old text
    ok = ("#error", "#define MKX_EARLY_XROWS 1")
    for src in (xa, xc):
        code = [x.split("//")[0] for x in src.split("\n")]
        bad = [x for x in code if "MKX_EARLY" in x and "defined(MKX_EARLY" not in x and not x.lstrip().startswith(ok)]
        assert not bad, bad


@contextlib.contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host file runs with ``--noconftest`` (the root conftest imports ttnn at collection)."""
    print(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        print(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


def _galaxy_defaults():
    """tt/galaxy_defaults.py (no ttnn import), loaded from its file"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("glm53_galaxy_defaults_host", GD_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_early_host_knob(monkeypatch):
    early = _host_fn("early_defines")
    monkeypatch.delenv("GLM53_MKX_EARLY", raising=False)
    assert early(True, True) == [], "the raw default (the galaxy default sets it: test_early_galaxy_default)"
    monkeypatch.setenv("GLM53_MKX_EARLY", "0")
    assert early(True, True) == []
    monkeypatch.setenv("GLM53_MKX_EARLY", "1")
    assert early(True, True) == [("MKX_EARLY", "1")]
    assert early(False, True) == [] and early(False, False) == [], "no y2 (the sparse_dsa MoE): inactive, no define"
    with expect_error(ValueError, "needs the batched expert TP"):
        early(True, False)
    for bad in ("2", "on", "true", ""):
        monkeypatch.setenv("GLM53_MKX_EARLY", bad)
        with expect_error(ValueError, "0 or 1"):
            early(True, True)
    src = MKX.read_text()
    # xa / xc only, after MKX_Y2_INIT; xw's defines (its JIT hash) and MKX_REV never change
    i = src.index("edefs = early_defines(y2 is not None, self.xsl is not None)")
    add = src.index("adefs += edefs")
    assert src.index("if raw_sync():") < i < src.index('cdefs.append(("MKX_Y2_INIT", "1"))') < add
    assert "\n            cdefs += edefs\n" in src[add : add + 200]
    assert "wdefs += edefs" not in src and src.count("edefs") == 4
    assert re.search(r"^MKX_REV = 24 ", src, re.M)


def _protect_env(monkeypatch, keys) -> None:
    """keys absent now and restored after the test (galaxy_defaults.apply writes os.environ directly)"""
    for k in keys:
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)


def test_early_galaxy_default(monkeypatch):
    """GLM53_MKX_EARLY is a galaxy default: apply() sets it on the 8x4 when unset; an explicit 0 stays 0 (no define);
    other meshes get nothing"""
    GD = _galaxy_defaults()
    early = _host_fn("early_defines")
    keys = list(GD.KNOBS) + ["GLM53_MODEL_TRACE_MB", "GLM53_PF_SPARSE_T", "GLM53_PREFILL_BUCKETS"]
    assert GD.KNOBS["GLM53_MKX_EARLY"] == "1"
    _protect_env(monkeypatch, keys)
    got = GD.apply((8, 4), 1048576)
    assert got["GLM53_MKX_EARLY"] == "1" == os.environ["GLM53_MKX_EARLY"]
    assert early(True, True) == [("MKX_EARLY", "1")] and early(False, True) == []
    _protect_env(monkeypatch, keys)
    monkeypatch.setenv("GLM53_MKX_EARLY", "0")
    got = GD.apply((8, 4), 1048576)
    assert "GLM53_MKX_EARLY" not in got and os.environ["GLM53_MKX_EARLY"] == "0" and early(True, True) == []
    _protect_env(monkeypatch, keys)
    assert GD.apply((2, 4), 1048576) == {} and "GLM53_MKX_EARLY" not in os.environ


def test_early_xrows_source():
    """MKX_EARLY_XROWS (GLM53_MKX_EARLY with MKX_XROWS_COAL in a megakernel composite): xrows_early() issues the row
    reads through the expert phase's own helper (xrows_read) into CB_X's base before the hand-off wait of the reader
    phase ahead of the experts (reader_a.cpp MK_BEFORE_WAIT_HOOK at MK_SPLIT - 1, wired by reader_moe_a.cpp under the
    same macro); the expert phase then only retires them and pushes CB_X"""
    xa, ra, rma = XA.read_text(), RA.read_text(), RMA.read_text()
    k = xa.index("void kernel_main() {")
    h = xa.index("FORCE_INLINE void xrows_read(const XAcc& x, uint32_t xl1, uint32_t xpage) {")
    d = xa.index("#if defined(MKX_EARLY) && defined(MKX_XROWS_COAL)\n")
    assert (
        h < xa.index("#if defined(MKX_XROWS) && defined(MKX_XROWS_PRE)") < d < xa.index("#define MKX_EARLY_XROWS 1") < k
    )
    e0 = xa.index("FORCE_INLINE void xrows_early() {")
    early = xa[e0 : xa.index("\n}\n", e0) + 1]
    order = ["mk::cb_rewind(CB_X);", "const uint32_t xl1 = get_write_ptr(CB_X), xpage = get_tile_size(CB_X);",
             "noc_async_read_barrier();  // the prologue's zeros land before the row reads into the same tiles",
             "xrows_read<KT_GU>(x, xl1, xpage);", "MK_MARK();"]  # fmt: skip
    pos = [early.index(s) for s in order]
    assert pos == sorted(pos) and "cb_push_back" not in early and early.count("noc_async_read_barrier();") == 1
    assert xa.count("xrows_read<KT_GU>(x, xl1, xpage);") == 2 and xa.count("noc_async_read(x.get_noc_addr(k, ") == 5
    # the expert phase: MKX_EARLY_XROWS skips the zero-fill barrier + the reads, keeps the retiring barrier + the push
    on = xa.index("#if defined(MKX_EARLY_XROWS)\n        // GLM53_MKX_EARLY: xrows_early()", k)
    end = xa.index("#endif  // MKX_EARLY_XROWS\n        noc_async_read_barrier();\n        cb_push_back(CB_X, KT_GU);")
    assert on < xa.index("xrows_read<KT_GU>(x, xl1, xpage);", k) < end and "(void)xl1;" in xa[on:end]
    # the hook: before phase P's hand-off wait only, inert without MK_BEFORE_WAIT_PHASE; wired under MKX_EARLY_XROWS
    hk = ra.index("#if defined(MK_BEFORE_WAIT_PHASE)\n    if constexpr (P == MK_BEFORE_WAIT_PHASE) {")
    assert ra.index("FORCE_INLINE void move_phase() {") < hk < ra.index("mk::wait_handoff(WAIT_SEM, WAIT_COUNT);")
    assert ra.count("MK_BEFORE_WAIT") == 3 and "#if defined(MK_BEFORE_WAIT" not in xa
    w = rma.index("#if defined(MKX_EARLY_XROWS)\n")
    blk = rma[w : rma.index("namespace dense {")]
    assert rma.index("namespace experts {") < w and rma.index("#define MKX_XROWS_PRE 1") < w
    for s in ('#error "MKX_EARLY_XROWS', "static_assert(MK_SPLIT >= 1,", "static_assert(MK_SPLIT - 1 > MK_AFTER_PHASE,",
              "#define MK_BEFORE_WAIT_PHASE (MK_SPLIT - 1)", "#define MK_BEFORE_WAIT_HOOK() ::experts::xrows_early()",
              "#if defined(MKX_EARLY_XROWS) && !defined(MK_BEFORE_WAIT_PHASE)"):  # fmt: skip
        assert s in blk, s


def test_early_x_is_final():
    """the hook's x reads go out at phase MK_SPLIT - 1 (shared down with fold_y2), after the router phase (3) on the same
    RISC waited for SEM_HN's count = every hn2 producer: the FFN mhc_pre's GRID_X x GRID_Y K cores each signal every
    reader once (mhc_pre.py MK_SIGNAL_N = the readers); the attention's mhc_pre signals the same semaphore for hn, waited
    by phase 0 with the same count; x of the experts is hn2"""
    ms, mp = (TT / "mk_sparse.py").read_text(), (TT / "mhc_pre.py").read_text()
    g = re.search(r"^GRID_X = GRID_Y = (\d+)  # K cores", mp, re.M)
    producers = int(g.group(1)) ** 2
    assert producers == 64
    assert 'k_defines += [("MK_SIGNAL_SEM_ID", str(sem_id)), ("MK_SIGNAL_N", str(len(targets)))]' in mp
    p0 = ms.index("        phases = [\n")
    ph = ms[p0 : ms.index("        ]\n", p0)]
    firsts = [ph.index(s) for s in ('lins["kda_qkv"]', 'lins["kda_lowrank"]', 'lins["kda_o"]',
                                    "self._router_ds(router)", "shared.ds_gate_up", "shared.ds_down")]  # fmt: skip
    assert firsts == sorted(firsts)
    router = ph[firsts[3] : firsts[4]]
    assert f"wait=(SEM_HN, {producers})," in router and f"wait=(SEM_HN, {producers})," in ph[firsts[0] : firsts[1]]
    assert ph.count("SEM_HN") == 2  # phases 0 (hn) and 3 (hn2): nothing else consumes the count in between
    assert "sig = (SEM_HN, rd.noc_coords())" in ms
    assert re.search(r"mf\.parts\(s1, hn2, [^)]*signal=sig", ms) and re.search(r"ma\.parts\(x, hn, ", ms)
    assert 'defs_ += [("MK_SPLIT", "6" if self.fold_y2 else "5"), ' in ms  # the hook's phase 5 > the router's 3
    assert "self.experts.parts_raw(hn2, ids, scores, y, t=t," in ms


# ---------------------------------------------------------------------------------------------- GLM53_MHC_PRE_SPLIT
def test_pre_split_galaxy_default(monkeypatch):
    """GLM53_MHC_PRE_SPLIT is a galaxy default: apply() sets it on the 8x4 when unset; an explicit 0 stays 0; other
    meshes get nothing"""
    GD = _galaxy_defaults()
    keys = list(GD.KNOBS) + ["GLM53_MODEL_TRACE_MB", "GLM53_PF_SPARSE_T", "GLM53_PREFILL_BUCKETS"]
    assert GD.KNOBS["GLM53_MHC_PRE_SPLIT"] == "1"
    _protect_env(monkeypatch, keys)
    got = GD.apply((8, 4), 1048576)
    assert got["GLM53_MHC_PRE_SPLIT"] == "1" == os.environ["GLM53_MHC_PRE_SPLIT"]
    _protect_env(monkeypatch, keys)
    monkeypatch.setenv("GLM53_MHC_PRE_SPLIT", "0")
    got = GD.apply((8, 4), 1048576)
    assert "GLM53_MHC_PRE_SPLIT" not in got and os.environ["GLM53_MHC_PRE_SPLIT"] == "0"
    _protect_env(monkeypatch, keys)
    assert GD.apply((2, 4), 1048576) == {} and "GLM53_MHC_PRE_SPLIT" not in os.environ


# stack_model / mhc_pre import ttnn at module level: the child stubs it (this process stays ttnn-free); every case starts
# a new "process": the GLM53_ environment cleared, the launch latch (_PRE_SPLIT_LAUNCH) and the logged kinds reset
_PRE_SPLIT_CHILD = r"""
import contextlib, inspect, io, os, sys, types
from unittest import mock

import torch

sys.modules["ttnn"] = mock.MagicMock()
from models.autoports.zai_org_glm_5_3_flash.tt import galaxy_defaults as GD
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as MP
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as STM
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_DSA, SPARSE_KDA

KINDS, NAMES, SEQ = (False, True, "dsa"), {False: DENSE_KDA, True: SPARSE_KDA, "dsa": SPARSE_DSA}, 1 << 20
_ITEMS, PARSED = GD.knob_set_items, []  # GLM53_STACK_KNOB_SET parses: the calling function of each


def _items(spec=None):
    PARSED.append(sys._getframe(1).f_code.co_name)
    return _ITEMS(spec)


GD.knob_set_items = _items


def launch(env=None, mesh=(8, 4)):
    for k in [k for k in os.environ if k.startswith("GLM53_")]:
        del os.environ[k]
    os.environ.update(env or {})
    STM._PRE_SPLIT_LAUNCH.clear()
    STM._PRE_SPLIT_LOGGED.clear()
    return GD.apply(mesh, SEQ)


def splits(kinds, tp=32, t=16):
    out, buf = [], io.StringIO()
    with contextlib.redirect_stdout(buf):
        for k in kinds:
            STM.set_knobs(k, tp, t)
            out.append(MP.MhcPre._pre_split())
    return out, buf.getvalue()


for t in (1, 4, 16):  # the 8x4 launch: every kind, in any order, at every width; one line per kind
    launch()
    got, log = splits(KINDS + KINDS[::-1], 32, t)
    assert got == [True] * 6 and os.environ["GLM53_MHC_PRE_EARLY"] == "1", (t, got)
    for k in KINDS:
        assert log.count(f"GLM53_MHC_PRE_SPLIT active ({NAMES[k]}:") == 1, (t, k, log)
launch({"GLM53_MHC_PRE_SPLIT": "0"})  # an explicit 0: every kind off, also once set_knobs popped it
got, log = splits(KINDS + KINDS[::-1])
assert got == [False] * 6 and "GLM53_MHC_PRE_SPLIT" not in os.environ and "active" not in log, (got, log)
assert launch(mesh=(2, 4)) == {}  # other meshes (no galaxy defaults): TP8 and one chip off
g8, l8 = splits(KINDS, 8)
g1, l1 = splits(KINDS, 1)
assert g8 == g1 == [False] * 3 and "active" not in l8 + l1, (g8, g1)
launch()  # the one-chip tables (no PRE_EARLY) never split
assert splits(KINDS, 1)[0] == [False] * 3
launch({"GLM53_STACK_KNOB_SET": "s:GLM53_MHC_PRE_SPLIT=0"})  # a per-kind override, without leaking to other kinds
assert splits((False, True, "dsa", True, False))[0] == [True, False, True, False, True]
launch({"GLM53_MHC_PRE_SPLIT": "0", "GLM53_STACK_KNOB_SET": "a:GLM53_MHC_PRE_SPLIT=1"})
assert splits(KINDS)[0] == [False, False, True]
# one parser: the galaxy defaults' KDA-kind reads and set_knobs' overrides both parse through knob_set_items (neither
# splits the items itself); "," separates items when no ";" appears, "K=-" drops K for that kind, ";" keeps a value's
# commas; an unknown kind letter is ignored by the KDA-kind reads and set_knobs' KeyError
assert {"set_knobs", "_kda_kind_values"} <= set(PARSED), PARSED
assert all('split(":", 1)' not in inspect.getsource(f) for f in (STM.set_knobs, GD._kda_kind_values))
launch({"GLM53_STACK_KNOB_SET": "a:GLM53_MHC_PRE_SPLIT=-,d:GLM53_MHC_PRE_SPLIT=0"})
assert splits(KINDS)[0] == [False, True, False]
assert list(_ITEMS("d:A=1,,s:B=-")) == [("d", "A", "1"), ("s", "B", "-")]
assert list(_ITEMS("d:A=1,2;s:B=3;")) == [("d", "A", "1,2"), ("s", "B", "3")]
launch({"GLM53_STACK_KNOB_SET": "x:GLM53_MK_OPT=O3"})
try:
    STM.set_knobs(False, 32, 16)
    raise RuntimeError("set_knobs took an unknown kind")
except KeyError:
    pass
launch()  # read once per process: a later environment change is ignored
splits((True,))
os.environ["GLM53_MHC_PRE_SPLIT"] = "0"
assert splits(KINDS)[0] == [True] * 3
launch({"GLM53_MHC_PRE_SPLIT": "extras"})  # a foreign value (the mesh model's old name): off, one warning
got, log = splits(KINDS + KINDS)
assert got == [False] * 6 and log.count("WARNING: GLM53_MHC_PRE_SPLIT='extras'") == 1, log
# GLM53_KDA_VERIFY_PIPE / _FLAT are no per-kind knobs (set_knobs leaves them in the environment): a sparse_dsa
# GLM53_STACK_KNOB_SET entry for them reaches the KDA builds after it, so the galaxy defaults drop FLAT and RSPLIT
# and the sparse_kda verify's KdaPre (8 user groups at TP 32: 2 heads x kda_pre spread 4) builds knob-off; the RSPLIT
# default left on beside that entry's values is the refusal
from models.autoports.zai_org_glm_5_3_flash.tt import shard_plan as SP
from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre

VP, VF, VR = "GLM53_KDA_VERIFY_PIPE", "GLM53_KDA_VERIFY_FLAT", "GLM53_KDA_VERIFY_RSPLIT"
assert set(GD._PER_KIND_KEYS) <= set(STM.KNOB_KEYS) and not {VP, VF} & set(STM.KNOB_KEYS), GD._PER_KIND_KEYS
TTNN = sys.modules["ttnn"]
TTNN.get_device_tensors.side_effect = lambda t: [t]
TTNN.to_torch.side_effect = lambda t: t


def verify_rows():  # the sparse_kda verify build's row groups (KdaPre reads its knobs at construction)
    kda = types.SimpleNamespace(device=None, h=2, kda_pre_spread=4, kda_rec_spread=2, kda_rec_ugroups=8,
                                kda_verify_sel=object(), kda_verify_groups=8, taps_hist=torch.zeros(3, 768),
                                tap_cur=torch.zeros(1, 768))  # fmt: skip
    return KdaPre(kda).vrsplit


for spec, rows in ((None, 8), ("a:GLM53_KDA_VERIFY_PIPE=0;a:GLM53_KDA_VERIFY_FLAT=0", 1)):
    got = launch({"GLM53_STACK_KNOB_SET": spec} if spec else None)
    splits((False, "dsa", True))
    assert verify_rows() == rows, spec
    assert (VF in got, VR in got) == (rows > 1, rows > 1), (spec, got)
os.environ[VR] = "1"
try:
    verify_rows()
    raise RuntimeError("KdaPre took GLM53_KDA_VERIFY_RSPLIT=1 without PIPE / FLAT")
except ValueError as e:
    assert "needs GLM53_KDA_VERIFY_PIPE=1 and GLM53_KDA_VERIFY_FLAT=1" in str(e), e
# the RSPLIT lanes check in every owner rewrite (tt/tp8_layer.py TPDenseKDALayer._hooked, tt/tp8_sparse.py
# TPSparseKDALayer._hooked_sparse, tt/tp_stack.py TPStack / TPSparseStack.layer_hook + finalize) on that builder's own
# values: the P6 wait it wrote, the semaphore and producer count it hands its lanes (fabric_ar lane_parts), its o_proj
# readers' cores and MK_FAB_MASK. Captured programs stand in for MkLayer's / MkSparseLayer's (their builds place cores
# on a device): every check call is recorded
from models.autoports.zai_org_glm_5_3_flash.tt import tp8_layer as TPL
from models.autoports.zai_org_glm_5_3_flash.tt import tp8_sparse as TPSP
from models.autoports.zai_org_glm_5_3_flash.tt import tp_stack as TPS
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import SEM_A

NR, NS, CHECKS, LANES = 16, types.SimpleNamespace, [], []
_LANES_CHECK = SP.kda_verify_rsplit_lanes_check
SP.kda_verify_rsplit_lanes_check = lambda *a: CHECKS.append(a) or _LANES_CHECK(*a)


def kern(src, rects, defines):  # a captured KernelDescriptor: the fields the owner rewrites read and write
    rt = {}
    for x0, y0, x1, y1 in rects:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                rt.setdefault(x, {})[y] = [0]
    crs = NS(ranges=lambda: [NS(start=NS(x=a, y=b), end=NS(x=c, y=d)) for a, b, c, d in rects])
    return NS(kernel_source="models/autoports/zai_org_glm_5_3_flash/tt/kernels/" + src, core_ranges=crs,
              runtime_args=rt, defines=list(defines), compile_time_args=[0] * 64)  # fmt: skip


def captured(sparse, rsplit):  # the o_proj readers (NR cores), the owners, the sparse kind's weight streamer
    own = [("MK_POST_WAIT_SEM", str(SEM_A)), ("MK_POST_WAIT_COUNT", str(NR))] + [("KDA_VERIFY_RSPLIT", "1")] * rsplit
    if not sparse:
        return NS(kernels=[kern("mk/reader_a.cpp", [(0, 8, 7, 9)], []), kern("mk/owner_dm0.cpp", [(0, 0, 7, 7)], own)],
                  semaphores=[], cbs=[])  # fmt: skip
    return NS(kernels=[kern("mk/reader_moe_a.cpp", [(0, 8, 5, 9)], []), kern("mk/reader_moe_a.cpp", [(6, 8, 7, 9)], []),
                       kern("mk/reader_moe_w.cpp", [(0, 10, 7, 10)], [("MK_X_CT", "0")]),
                       kern("mk/owner_dm0.cpp", [(0, 0, 7, 7)], own + [("MK_POST2_WAIT_COUNT", str(NR))])],
              semaphores=[], cbs=[])  # fmt: skip


def build(stack, sparse, *, rsplit=True, lanes=2, nr=NR):  # -> (check calls, lane_parts (cons_sem, readers) calls)
    del CHECKS[:], LANES[:]
    def lane_parts(r, c, iters, readers, owner_noc, cons_sem):  # fabric_ar FabricPhase.lane_parts: the lanes
        LANES.append((cons_sem, readers))
        return [], []

    fab = NS(lanes=[(1, 1)] * lanes, brows=1, producer_args=lambda: [1], consumer_args=lambda: [2], io=lambda: [],
             add_connections=lambda *a, **k: None, lane_parts=lane_parts)  # fmt: skip
    desc = captured(sparse, rsplit)
    if stack:  # spec_mtp's verify widths and stack_model's main stack
        stk = object.__new__(TPS.TPSparseStack if sparse else TPS.TPStack)
        stk.fl = NS(fabric=fab, rev_defs=[], n_readers=nr, R=1, C=1, owner_noc=[], extra_sems=[])
        stk.n, stk.info = 1, {}
        stk.layer_hook(desc)
        stk.finalize([], desc)
    else:  # the per-layer launches
        lay = object.__new__(TPSP.TPSparseKDALayer if sparse else TPL.TPDenseKDALayer)
        lay.__dict__.update(fabric=fab, rev_defs=[], n_readers=nr, R=1, C=1, owner_noc=[], extra_sems=[], xsl=False,
                            placement=None)  # fmt: skip
        if sparse:
            lay._hooked_sparse([], desc)
        else:
            lay._hooked([], desc, fab_mask=TPS.TPStack.FAB_MASK, cons_sem=SEM_A, iters=2)
    return list(CHECKS), list(LANES)


for stack in (False, True):
    for sparse, mask in ((False, TPS.TPStack.FAB_MASK), (True, TPS.TPSparseStack.FAB_MASK)):
        chk, lanes_got = build(stack, sparse)
        assert chk == [((SEM_A, 2), SEM_A, 2, NR, NR, mask)], (stack, sparse, chk)
        assert lanes_got == [(chk[0][1], chk[0][3])], (stack, sparse, lanes_got)  # the lanes' semaphore + producers
        assert build(stack, sparse, rsplit=False)[0] == [], (stack, sparse)
        for kw, msg in ((dict(nr=NR - 1), "producers"), (dict(lanes=3), "P6 waits")):
            try:
                build(stack, sparse, **kw)
                raise RuntimeError(f"the RSPLIT lanes check passed {kw}")
            except AssertionError as e:
                assert msg in str(e), (stack, sparse, kw, e)
SP.kda_verify_rsplit_lanes_check = _LANES_CHECK
print("PRE_SPLIT_OK")
"""


def test_pre_split_set_knobs():
    """stack_model.set_knobs on the 8x4 launch, for GLM53_MHC_PRE_SPLIT and GLM53_KDA_VERIFY_RSPLIT. MHC_PRE_SPLIT on
    every TP kind (dense_kda, sparse_kda, sparse_dsa) at t 1 / 4 / 16 in any kind order, one "active" line per kind; an explicit 0 keeps every kind off; TP8 / one chip
    off; GLM53_STACK_KNOB_SET overrides it per kind without leaking (set_knobs and the galaxy defaults parse the
    variable with galaxy_defaults.knob_set_items); the launch value is read once per process; a foreign value warns
    once and stays off. GLM53_KDA_VERIFY_RSPLIT in the same child: a sparse_dsa GLM53_STACK_KNOB_SET entry for
    GLM53_KDA_VERIFY_PIPE / _FLAT (no per-kind knobs) drops the FLAT / RSPLIT defaults, so the sparse_kda verify's
    KdaPre after set_knobs builds knob-off instead of refusing; the four TP owner rewrites (tt/tp8_layer.py,
    tt/tp8_sparse.py, tt/tp_stack.py) pass the lanes check their own P6 wait, lane semaphore, lane producers and
    o_proj readers, the same semaphore and producers their lanes get"""
    root = Path(__file__).resolve().parents[4]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_")}
    env["PYTHONPATH"] = str(root)
    r = subprocess.run([sys.executable, "-c", _PRE_SPLIT_CHILD], cwd=str(root), env=env, capture_output=True, text=True,
                       timeout=300)  # fmt: skip
    assert r.returncode == 0 and "PRE_SPLIT_OK" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-4000:])
    assert "ttnn" not in sys.modules


def test_pre_split_knob_names():
    """the stack's GLM53_MHC_PRE_SPLIT (0 / 1: mhc_pre.py, stack_model, galaxy_defaults) and the mesh model's pre-block
    split GLM53_MESH_MHC_PRE_SPLIT (cols / extras: tt/mhc.py, the custom-op test) never meet: no mesh-model file reads the
    stack's name"""
    demos = Path(__file__).resolve().parents[1]
    mesh = (demos / "tt" / "mhc.py").read_text()
    assert 'os.environ.get("GLM53_MESH_MHC_PRE_SPLIT", "extras")' in mesh
    op_test = (demos / "tests" / "custom_ops" / "test_mhc_pre_block.py").read_text()
    assert 'os.environ.get("GLM53_MESH_MHC_PRE_SPLIT", "cols")' in op_test
    readers = sorted(str(p.relative_to(demos)) for p in (demos / "tt").rglob("*.py")
                     if '"GLM53_MHC_PRE_SPLIT"' in p.read_text())  # fmt: skip
    assert readers == [], readers
    assert 'os.environ.get("GLM53_MHC_PRE_SPLIT")' in (TT / "mhc_pre.py").read_text()
    assert '"GLM53_MHC_PRE_SPLIT": "1"' in (TT / "galaxy_defaults.py").read_text()


# ---------------------------------------------------------------------------------------------- GLM53_MHC_RMS_EARLY
MHC_K = TT / "kernels" / "mhc_pre"


def test_rms_early_kernel_order():
    """MHC_RMS_EARLY in the root's sync role: the compact coefficient tile is written and barriered BEFORE the round-2
    fragment (whose invh write lands before the readers' MK_INVH_SEM increments), and the post / comb tiles come after;
    knob off, the fragment runs where round 2 always ran (after the post / comb / coefficient tiles and CB_COEF's pop)
    """
    sync = (MHC_K / "sync.cpp").read_text()
    frag = (MHC_K / "mhc_pre_round2.inc").read_text()
    inc = '#include "mhc_pre_round2.inc"'
    assert sync.count(inc) == 2
    on = sync.index("#if defined(MHC_RMS_EARLY)\n        // GLM53_MHC_RMS_EARLY")
    first, second = sync.index(inc), sync.rindex(inc)
    coef_w = sync.index("noc_async_write_page(0, coef_out, scr);\n        noc_async_write_barrier();", on)
    assert on < coef_w < first < sync.index("if constexpr (PC_MODE == 0) {", first)
    assert "ORDERING INVARIANT" in sync[on:first] and "mhc_post reads coef without a hand-off" in sync[on:first]
    # knob off: the old compact-coefficient write after post / comb, the fragment after CB_COEF's pop
    off_coef = sync.index("#if !defined(MHC_RMS_EARLY)\n        // compact coefficients for the fused post-mix")
    assert sync.index("if constexpr (PC_MODE == 0) {", first) < off_coef
    pop = sync.index("cb_pop_front(CB_COEF, 1);", off_coef)
    assert pop < sync.index("#if !defined(MHC_RMS_EARLY)\n" + inc, pop) == second - len("#if !defined(MHC_RMS_EARLY)\n")
    assert re.search(r"#if defined\(MHC_RMS_EARLY\) && defined\(MHC_COEF_PLANE\)\n#error", sync)
    # the fragment: round 2's gather, 1/rms(h) ready, the invh tile before the readers' increments, the leaders' message
    order = ["noc_semaphore_wait(sem_ptr(SEM_G2), N_GROUPS);", "cb_wait_front(CB_INVR, 1);",
             "noc_async_write_page(0, invh_out, get_read_ptr(CB_INVR));", "get_semaphore(MK_INVH_SEM)",
             "get_semaphore(SEM_M2)", "cb_pop_front(CB_INVR, 1);"]  # fmt: skip
    pos = [frag.index(x) for x in order]
    assert pos == sorted(pos), pos
    assert "MK_MARK();  // 4" not in frag  # (mark 4 = 1/rms(h) out: after the fragment, either order)


def _mhc_pre_fns(*names):
    """mhc_pre.py's module-level names exercised from its source (importing the module would import ttnn)"""
    src = (TT / "mhc_pre.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "_RMS_EARLY_LOGGED")]  # fmt: skip
    ns = {"os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "mhc_pre.py", "exec"), ns)  # noqa: S102 (our own source)
    return ns


def test_rms_early_host_knob(monkeypatch, capsys):
    """rms_early_defines: the define only with the knob, a cp build with the knob refused; descriptor() adds it to the root sync
    role alone; rms_early_on reads GLM53_MHC_RMS_EARLY (default off) and logs once; MkMla (both mhc calls) and MkMoE
    pass it, the KDA stacks (MkLayer / MkSparseLayer) do not"""
    ns = _mhc_pre_fns("rms_early_defines", "rms_early_on")
    d = ns["rms_early_defines"]
    assert d(False, []) == [] and d(True, []) == [("MHC_RMS_EARLY", "1")] and d(False, [("MHC_COEF_PLANE", "1")]) == []
    refused = False  # a cp build with the knob: refused (no expect_error fixture under --noconftest)
    try:
        d(True, [("MHC_COEF_PLANE", "1")])
    except ValueError as e:
        refused = "cp build" in str(e)
    assert refused
    monkeypatch.delenv("GLM53_MHC_RMS_EARLY", raising=False)
    assert ns["rms_early_on"]() is False and "active" not in capsys.readouterr().out
    monkeypatch.setenv("GLM53_MHC_RMS_EARLY", "1")
    assert ns["rms_early_on"]() and ns["rms_early_on"]()
    assert capsys.readouterr().out.count("GLM53_MHC_RMS_EARLY active") == 1
    src = (TT / "mhc_pre.py").read_text()
    assert 'root_defines = [("MHC_ROOT", "1")] + extra + rms_early_defines(rms_early, extra)' in src
    assert src.count("rms_early_defines(") == 2  # (its definition + the root sync role)
    assert (TT / "mk_mla.py").read_text().count("rms_early=_mhc.rms_early_on()") == 2
    assert (TT / "mk_moe.py").read_text().count("rms_early=_mhc.rms_early_on()") == 1
    for f in ("mk_layer.py", "mk_sparse.py", "mk_dense.py"):
        assert "rms_early" not in (TT / f).read_text(), f


def test_rms_early_galaxy_default(monkeypatch):
    """GLM53_MHC_RMS_EARLY is a galaxy default: apply() sets it on the 8x4 when unset; an explicit 0 stays 0 (the
    knob-off kernels); other meshes get nothing"""
    GD = _galaxy_defaults()
    keys = list(GD.KNOBS) + ["GLM53_MODEL_TRACE_MB", "GLM53_PF_SPARSE_T", "GLM53_PREFILL_BUCKETS"]
    assert GD.KNOBS["GLM53_MHC_RMS_EARLY"] == "1"
    on = _mhc_pre_fns("rms_early_on")["rms_early_on"]
    _protect_env(monkeypatch, keys)
    got = GD.apply((8, 4), 1048576)
    assert got["GLM53_MHC_RMS_EARLY"] == "1" == os.environ["GLM53_MHC_RMS_EARLY"] and on()
    _protect_env(monkeypatch, keys)
    monkeypatch.setenv("GLM53_MHC_RMS_EARLY", "0")
    got = GD.apply((8, 4), 1048576)
    assert "GLM53_MHC_RMS_EARLY" not in got and os.environ["GLM53_MHC_RMS_EARLY"] == "0" and not on()
    _protect_env(monkeypatch, keys)
    assert GD.apply((2, 4), 1048576) == {} and "GLM53_MHC_RMS_EARLY" not in os.environ and not on()


def test_zz_no_ttnn_was_imported_by_the_suite():
    """The unit CI runs this file without the root conftest: a host suite must never pull ttnn in (it would open
    every chip of the box beside a running mesh job)."""
    import sys

    assert "ttnn" not in sys.modules
