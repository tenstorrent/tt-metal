# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts' multi-layer scratch race and MKX_RAW_SYNC (GLM53_MKX_RAW_SYNC=1, opt-in), host only (no ttnn, no
device). In a stack program (MK_LAYERS > 1) xa.cpp of layer l works in place on its core's xw.cpp scratch after
MKX_SEM_LIST: CB_RAW (MKX_RW_XA: sel / rank / the weights tile) and the list page's weight rows (the scatter, the
routing weight tiles); neither page is flow-controlled against xa (xc pops the list right after reading n, CB_RAW is
never pushed), and xw of layer l + 1 issues writes into both before its router-tail hand-off.

* the byte map (mkx_common.hpp's constants; the 8x4 XSL layout and the one-chip EP one): xw(l + 1)'s pre-hand-off
  writes into CB_RAW (the table row, the live page) miss every byte xa touches there; its list-page zero fill covers
  every weight row xa scatters / reads -> the colliding write;
* an exhaustive interleaving model of one reader core (weight streamer: dense phases + xw; compute: dense phases + xc;
  activation mover: xa) over 1-3 stacked layers: without the hand-back an expert-only reader (it sits out every dense
  phase) whose expert stream fits its ring lets xw(l + 1) write while xa(l) still reads -- with MKX_RW_XA on AND off
  (off only shortens the window in time) -- while a reader gated by its dense phases, a long stream or a one-layer
  program cannot; with the hand-back no interleaving does and none deadlocks;
* the kernel sources (where the wait and the reset sit) and the host knob (off by default; xw / xa defines only).

    pytest --noconftest models/demos/glm53_flash/tests/test_mkx_raw_sync_host.py -q
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

TT = Path(__file__).resolve().parents[3] / "autoports" / "zai_org_glm_5_3_flash" / "tt"
XW = TT / "kernels" / "mkx" / "xw.cpp"
XA = TT / "kernels" / "mkx" / "xa.cpp"
COMMON = TT / "kernels" / "mkx" / "mkx_common.hpp"
MKX = TT / "mk_experts.py"
TILE16 = 2048  # a bf16 tile page (the ids tile, the weights tile)
MEM_ZEROS_SIZE = 512  # Blackhole dev_mem_map.h: the NoC zero-fill granule
TOPK = 8
E = 288  # routed experts (MAX_EXPERTS)


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


def test_scratch_byte_map():
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


def explore(**kw) -> dict:
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


def test_race_without_the_hand_back():
    """knob off (903900a212f's protocol): the race is reachable exactly where the device saw it"""
    base = dict(sync=False, ring=2)
    for rw_xa in (True, False):
        # an expert-only reader (dense 0) whose stream can end before xa's last weight-tile read: reachable with the
        # scatter on xa AND on xw (MKX_RW_XA off narrows the window in time, it does not close it)
        for g in (SHORT, TWO):
            assert explore(layers=2, rw_xa=rw_xa, dense=0, groups=g, **base)["viol"] > 0, (rw_xa, g)
        # ... not with a stream that ends only after a down block (= after xa's gather, its last access)
        r = explore(layers=2, rw_xa=rw_xa, dense=0, groups=LONG, **base)
        assert r["viol"] == 0 and r["dead"] == 0 and r["done"] > 0, (rw_xa, r)
        # a reader gated by its dense phases (more blocks than its ring: the next layer's compute needs P10(l))
        for g in (SHORT, PAST, LONG, TWO):
            r = explore(layers=2, rw_xa=rw_xa, dense=RD + 1, groups=g, **base)
            assert r["viol"] == 0 and r["dead"] == 0, (rw_xa, g, r)
        # one sparse layer per program: no next-layer xw
        for g in (SHORT, PAST):
            assert explore(layers=1, rw_xa=rw_xa, dense=0, groups=g, **base)["viol"] == 0
    # shards past the union read no weight tile: only the scatter (MKX_RW_XA) is exposed there
    assert explore(layers=2, rw_xa=True, dense=0, groups=PAST, **base)["viol"] > 0
    assert explore(layers=2, rw_xa=False, dense=0, groups=PAST, **base)["viol"] == 0


def test_hand_back_closes_every_interleaving():
    """MKX_RAW_SYNC: no violation, no deadlock, every run finishes -- 2 and 3 layers, RW_XA on / off, expert-only and
    gated readers, every stream shape, rings of 1 .. 3 slots"""
    for layers in (2, 3):
        for rw_xa in (True, False):
            for dense in (0, RD + 1):
                for g in (SHORT, PAST, LONG, TWO):
                    for ring in (1, 2, 3):
                        r = explore(layers=layers, sync=True, rw_xa=rw_xa, dense=dense, ring=ring, groups=g)
                        assert r["viol"] == 0 and r["dead"] == 0 and r["done"] > 0, (layers, rw_xa, dense, g, ring, r)


# ---- the sources ---------------------------------------------------------------------------------------------------
SYNC_IF = "#if defined(MKX_RAW_SYNC) && defined(MK_LAYERS) && MK_LAYERS > 1\n"


def _block(src: str, start: int) -> str:
    return src[start : src.index("#endif", start)]


def test_kernel_sources():
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


def _fn(name):
    tree = ast.parse(MKX.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MKX), "exec"), ns)  # noqa: S102 (our own source)
    return ns[name]


def test_host_knobs(monkeypatch):
    raw_sync = _fn("raw_sync")
    monkeypatch.delenv("GLM53_MKX_RAW_SYNC", raising=False)
    assert not raw_sync(), "opt-in"
    monkeypatch.setenv("GLM53_MKX_RAW_SYNC", "0")
    assert not raw_sync()
    monkeypatch.setenv("GLM53_MKX_RAW_SYNC", "1")
    assert raw_sync()
    src = MKX.read_text()
    split = src.index("wdefs, adefs = list(defs), list(defs)")
    guard = src.index("if raw_sync():")
    assert split < guard < src.index("wtail, atail = [], []"), "xw / xa defines only, after the w / a split"
    assert src[guard:].startswith(
        'if raw_sync():\n            wdefs.append(("MKX_RAW_SYNC", "1"))\n            adefs.append(("MKX_RAW_SYNC", "1"))\n'
    )
    assert src.count('"MKX_RAW_SYNC", "1"') == 2
    # MKX_RW_XA is opt-in since 2026-11-10 (the race above); no revision bump for either knob
    assert 'os.environ.get("GLM53_MKX_RW_XA", "0") == "1"' in src
    assert re.search(r"^MKX_REV = 24 ", src, re.M)
