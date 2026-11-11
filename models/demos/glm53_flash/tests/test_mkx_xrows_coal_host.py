# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts' activation mover / weight streamer (kernels/mkx/xa.cpp, xw.cpp), host only (no ttnn, no
device). Two subjects:

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

    pytest --noconftest models/demos/glm53_flash/tests/test_mkx_xrows_coal_host.py -q
"""

from __future__ import annotations

import ast
import os
import re
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
    coal, off = _branches(src)
    assert off == OLD_LOOP, "the knob-off branch must stay the 44adcb9ba40 per-row loop verbatim"
    assert src.count("#if defined(MKX_XROWS_COAL)") == 1
    # the block sits inside MKX_XROWS, between the zero-fill barrier and the push's barrier
    i = src.index("#if defined(MKX_XROWS_COAL)")
    assert src.index("#if defined(MKX_XROWS)\n    // decode rows only") < i < src.index("#elif defined(MKX_XPROG)")
    assert "the zeros land before the row reads" in src[i - 200 : i]
    assert src[src.index("#endif\n", i) + 7 :].lstrip().startswith("noc_async_read_barrier();")
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
