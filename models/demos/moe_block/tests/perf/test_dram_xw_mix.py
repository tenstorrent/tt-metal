# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DRAM read mix of a routed-expert call, nothing else: weights + gathered x rows, DRAM -> L1, one chip.

Question: can any placement / read size / NoC / layout download the weights AND x at >= 90% of 512 GB/s?
No compute, no forwarding, no tilize, no writes back - plain NoC reads into a ring of L1 slots.

Volumes per call (GLM-5.3 at Galaxy load, bfp4 weights 0.5625 B / elem):
  weights: E 8 x 3 x H 6144 x I 2048 = 169.9 MB; x: 1280 rows x 6144 bf16 (12 KB rows, row-major, one row a page,
  interleaved over the 8 banks), each expert's 160 rows a random subset (a random permutation split in 8).
K3 (E 28, H 3584, I 3072, M 91): weights 520.2 MB, x 2548 rows x 7 KB = 18.3 MB.

One kernel source serves every role: a continuous queue of DEPTH reads (one transaction id each) that issues nw weight
reads (page g = g0 + i * gstride: bank g % 8, offset (g / 8) * WSZ - pinned to a bank with gstride 8, rotating over
banks with gstride 1) and nx x reads (a per-core table of bank << 28 | offset entries read from DRAM before t0),
interleaved by ORDER (spread evenly / x first / weights first). Optional pacing holds x read j until t0 + j * gap.
Each kernel writes its wall-clock t0 / t1 (cycles) to DRAM; spans use 1350 MHz (the AI clock is logged).

Each variant runs three modes: weights alone, x alone, both (median of REPS). Results: log lines "DXW ..." and a
JSON line per variant in /localdev/mstaletovic/dram_xw/results.jsonl.
"""
import json
import os
import statistics
import threading
import time
import zlib
from dataclasses import asdict, dataclass

import pytest
import torch
from loguru import logger

import ttnn

pytestmark = pytest.mark.use_module_device

NB = 8
CLK_MHZ = 1350.0
REPS = 5
OUT_DIR = "/localdev/mstaletovic/dram_xw"

_K = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t DEPTH = get_compile_time_arg_val(0), WSZ = get_compile_time_arg_val(1);
constexpr uint32_t XSZ = get_compile_time_arg_val(2), SLOT = get_compile_time_arg_val(3);
constexpr uint32_t CB = get_compile_time_arg_val(4), ORDER = get_compile_time_arg_val(5);
constexpr uint32_t TAB_BYTES = get_compile_time_arg_val(6), BATCH = get_compile_time_arg_val(7);
static inline uint32_t wc() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
void kernel_main() {
    uint32_t a = 0;
    const uint32_t w_base = get_arg_val<uint32_t>(a++), x_base = get_arg_val<uint32_t>(a++);
    const uint32_t tab_base = get_arg_val<uint32_t>(a++), res_base = get_arg_val<uint32_t>(a++);
    const uint32_t idx = get_arg_val<uint32_t>(a++), nw = get_arg_val<uint32_t>(a++);
    const uint32_t g0 = get_arg_val<uint32_t>(a++), gstride = get_arg_val<uint32_t>(a++);
    const uint32_t nx = get_arg_val<uint32_t>(a++), gap = get_arg_val<uint32_t>(a++);
    const uint32_t l1 = get_write_ptr(CB);
    const uint32_t tab = l1 + DEPTH * SLOT, res = tab + TAB_BYTES;
    volatile tt_l1_ptr uint32_t* tp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    if (nx) {  // (this kernel's x read list: DRAM page idx of the table, before t0)
        noc_async_read(
            get_noc_addr_from_bank_id<true>(idx % NUM_DRAM_BANKS, tab_base + (idx / NUM_DRAM_BANKS) * TAB_BYTES),
            tab, TAB_BYTES);
        noc_async_read_barrier();
    }
    const uint32_t t0 = wc();
    uint32_t iw = 0, ix = 0;
    const uint32_t n = nw + nx;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t slot = i % DEPTH, trid = 1 + slot;
        if constexpr (BATCH) {  // (batches of DEPTH reads, a barrier between them)
            if (slot == 0 && i) {
                noc_async_read_barrier();
            }
        } else if (i >= DEPTH) {
            while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
            }
        }
        bool do_x;
        if (ix >= nx) {
            do_x = false;
        } else if (iw >= nw) {
            do_x = true;
        } else if constexpr (ORDER == 1) {
            do_x = true;
        } else if constexpr (ORDER == 2) {
            do_x = false;
        } else {
            do_x = ix * nw <= iw * nx;
        }
        noc_async_read_set_trid(trid);
        if (do_x) {
            if (gap) {
                while (wc() - t0 < ix * gap) {
                }
            }
            const uint32_t e = tp[ix++];
            noc_async_read(get_noc_addr_from_bank_id<true>(e >> 28, x_base + (e & 0x0FFFFFFF)), l1 + slot * SLOT, XSZ);
        } else {
            const uint32_t g = g0 + iw * gstride;
            ++iw;
            noc_async_read(
                get_noc_addr_from_bank_id<true>(g % NUM_DRAM_BANKS, w_base + (g / NUM_DRAM_BANKS) * WSZ),
                l1 + slot * SLOT, WSZ);
        }
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
    const uint32_t t1 = wc();
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res);
    r[0] = t0;
    r[1] = t1;
    r[2] = nw;
    r[3] = nx;
    noc_async_write(res, get_noc_addr_from_bank_id<true>(idx % NUM_DRAM_BANKS, res_base + (idx / NUM_DRAM_BANKS) * 64), 16);
    noc_async_write_barrier();
}
"""

MODELS = {
    # E, H, I, M (rows an expert), x rows total
    "glm": dict(E=8, H=6144, I=2048, M=160),
    "k3": dict(E=28, H=3584, I=3072, M=91),
}


@dataclass(frozen=True)
class Cfg:
    model: str = "glm"
    # weights: readers per bank next to the bank ("bank": optimal NOC0 core + row neighbours) or arbitrary mid cores
    w_cores: str = "bank"
    w_pb: int = 2  # readers per bank
    w_pat: str = "pin"  # pin: each reader streams its bank sequentially; rot: consecutive pages rotate over banks
    w_sz: int = 32768
    w_q: int = 4
    w_noc: int = 0
    # x: "sep" = own cores (BRISC, x_noc), "ncrisc" = the weight cores' NCRISC (NOC1), "wq" = in the weight queue
    x_mode: str = "sep"
    x_cores: str = "mid"  # sep only: mid (columns away from DRAM) / near (next to the bank's weight readers)
    x_n: int = 4
    x_sz: int = 2048  # bytes a read (a segment of a row; = row bytes for whole rows)
    x_q: int = 3
    x_noc: int = 1
    x_assign: str = "split"  # split: contiguous expert ranges a core (any bank); pin: a core reads only its bank's rows
    x_layout: str = "rows"  # rows: gathered rows; dense: pre-gathered, each bank's slab read sequentially
    order: int = 0  # wq: 0 spread, 1 x first, 2 weights first
    x_pace_us: float = 0.0  # sep: spread x reads over this many us (0 = as fast as possible)
    x_batch: bool = False  # sep: x reads in batches of x_q with a barrier between (instead of a continuous queue)
    # down weights as today's flat op: a third of the weight bytes read by dn_n cores as dn_sz blocks rotating over
    # the banks (the bank readers then stream only the gate/up two thirds); 0 = all weights bank-pinned
    dn_n: int = 0
    dn_sz: int = 9216
    dn_q: int = 2
    dn_cores: str = "rest"  # rest: away from the DRAM columns first; near: next to them first


V = {
    "base": Cfg(),
    # x read size / depth / count / NoC / placement
    "x4k": Cfg(x_sz=4096),
    "x12k": Cfg(x_sz=12288),
    "xq1": Cfg(x_q=1),
    "xq2": Cfg(x_q=2),
    "xq6": Cfg(x_q=6),
    "xq12": Cfg(x_q=12),
    "xn2": Cfg(x_n=2),
    "xn8": Cfg(x_n=8),
    "xn16": Cfg(x_n=16),
    "xnoc0": Cfg(x_noc=0),
    "xnear8": Cfg(x_cores="near", x_n=8),
    "xnear8pin": Cfg(x_cores="near", x_n=8, x_assign="pin"),
    "xnear16pin": Cfg(x_cores="near", x_n=16, x_assign="pin"),
    "xmid8pin": Cfg(x_n=8, x_assign="pin"),
    # x issued by the bank's own weight readers
    "xncrisc": Cfg(x_mode="ncrisc", x_assign="pin"),
    "xwq": Cfg(x_mode="wq", x_assign="pin"),
    "xwq_split": Cfg(x_mode="wq", x_assign="split"),
    "xwq_xfirst": Cfg(x_mode="wq", x_assign="pin", order=1),
    "xwq_wfirst": Cfg(x_mode="wq", x_assign="pin", order=2),
    "xwq12k": Cfg(x_mode="wq", x_assign="pin", x_sz=12288),
    # x paced over the weight time; x pre-gathered (dense)
    "xpace": Cfg(x_pace_us=360.0),
    "xpace_q6": Cfg(x_pace_us=360.0, x_q=6),
    "xdense": Cfg(x_layout="dense", x_sz=12288),
    "xdense_wq": Cfg(x_mode="wq", x_assign="pin", x_layout="dense", x_sz=12288),
    # weights
    "w8k": Cfg(w_sz=8192),
    "w16k": Cfg(w_sz=16384),
    "w64k": Cfg(w_sz=65536),
    "wq2": Cfg(w_q=2),
    "wq8": Cfg(w_q=8),
    "w1pb": Cfg(w_pb=1, w_q=8),
    "w3pb": Cfg(w_pb=3),
    "wrot": Cfg(w_pat="rot"),
    "wmid": Cfg(w_cores="mid"),
    "wnoc1": Cfg(w_noc=1),
    # weights as today's flat op: gate/up pinned by the bank readers, down 9 KB blocks rotating from 32 cores
    "wtoday": Cfg(dn_n=32),
    "wtoday_dq4": Cfg(dn_n=32, dn_q=4),
    "wtoday_near": Cfg(dn_n=24, dn_cores="near"),
    # x reads barrier-batched (issue x_q, wait for all) instead of a continuous queue
    "xbatch2": Cfg(x_batch=True, x_q=2),
    "xbatch3": Cfg(x_batch=True),
    "xbatch6": Cfg(x_batch=True, x_q=6),
    # combinations
    "xq2_n8": Cfg(x_q=2, x_n=8),
    "xnoc0_q6": Cfg(x_noc=0, x_q=6),
    "xnoc0_near8pin": Cfg(x_noc=0, x_cores="near", x_n=8, x_assign="pin"),
    "xncrisc_split": Cfg(x_mode="ncrisc", x_assign="split"),
    "wtoday_xq6": Cfg(dn_n=32, x_q=6),
    "wtoday_xwq": Cfg(dn_n=32, x_mode="wq", x_assign="pin"),
    "wtoday_dq4_xq6": Cfg(dn_n=32, dn_q=4, x_q=6),
    "wtoday_dq4_xwq12k": Cfg(dn_n=32, dn_q=4, x_mode="wq", x_assign="pin", x_sz=12288),
    # K3 volumes (7 KB x rows: 1792 B = a quarter row)
    "k3_base": Cfg(model="k3", x_sz=1792),
    "k3_xq2": Cfg(model="k3", x_sz=1792, x_q=2),
    "k3_xwq": Cfg(model="k3", x_sz=1792, x_mode="wq", x_assign="pin"),
    "k3_wtoday": Cfg(model="k3", x_sz=1792, dn_n=32),
    "k3_wtoday_xwq": Cfg(model="k3", x_sz=1792, dn_n=32, x_mode="wq", x_assign="pin"),
}


def _vol(m):
    p = MODELS[m]
    w = p["E"] * 3 * p["H"] * p["I"] * 9 // 16  # bfp4: 0.5625 B / elem
    return w, p["E"] * p["M"], p["H"] * 2


class _Clk:
    """Passive sysfs sampler of every card's AI clock (MHz) while a mode runs."""

    def __init__(self):
        self.paths = [
            f"/sys/class/tenstorrent/{d}/tt_aiclk"
            for d in sorted(os.listdir("/sys/class/tenstorrent"))
            if os.path.exists(f"/sys/class/tenstorrent/{d}/tt_aiclk")
        ]
        self.seen, self.per, self._stop = [], {}, threading.Event()

    def __enter__(self):
        def run():
            while not self._stop.is_set():
                for p in self.paths:
                    try:
                        v = int(open(p).read().strip())
                        self.seen.append(v)
                        lo, hi = self.per.get(p, (v, v))
                        self.per[p] = (min(lo, v), max(hi, v))
                    except (OSError, ValueError):
                        pass
                time.sleep(0.02)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def _placement(device, cfg):
    grid = device.compute_with_storage_grid_size()
    opt = [ttnn.CoreCoord(c.x, c.y) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    used = set()
    optset = {(c.x, c.y) for c in opt}

    def free(x, y):
        return 0 <= x < grid.x and 0 <= y < grid.y and (x, y) not in used and (x, y) not in optset

    # (mid: the 4 logical columns farthest from the banks' optimal reader columns)
    ocols = {c.x for c in opt}
    cols = sorted(range(grid.x), key=lambda x: (-min(abs(x - o) for o in ocols), x))[:4]
    mid = [(x, y) for y in range(grid.y) for x in sorted(cols)]
    w = []  # (core, bank)
    if cfg.w_cores == "bank":
        for b, c in enumerate(opt):
            mine = [(c.x, c.y)]
            for dx in (1, -1, 2, -2, 3, -3):
                if len(mine) == cfg.w_pb:
                    break
                if free(c.x + dx, c.y):
                    mine.append((c.x + dx, c.y))
            assert len(mine) == cfg.w_pb, f"bank {b}: only {mine}"
            used.update(mine)
            w += [(m, b) for m in mine]
    else:
        cand = [m for m in mid if m not in optset]
        for i in range(NB * cfg.w_pb):
            used.add(cand[i])
            w.append((cand[i], i % NB))
    x = []  # (core, bank or None)
    if cfg.x_mode == "sep":
        if cfg.x_cores == "near":
            per = cfg.x_n // NB
            for b, c in enumerate(opt):
                got = 0
                for dx, dy in ((0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 2), (0, -2), (2, 0), (-2, 0)):
                    if got == per:
                        break
                    if free(c.x + dx, c.y + dy):
                        used.add((c.x + dx, c.y + dy))
                        x.append(((c.x + dx, c.y + dy), b))
                        got += 1
                assert got == per, f"bank {b}: no free near cores"
        else:
            cand = [m for m in mid if m not in used and m not in optset]
            for i in range(cfg.x_n):
                used.add(cand[i])
                x.append((cand[i], i % NB))
    else:
        x = list(w)
    d = []  # down-weight cores (dn_n > 0): near = free cores next to the DRAM columns first, rest = the others
    if cfg.dn_n:
        ocols = sorted({c.x for c in opt})
        near_cols = {xx for o in ocols for xx in (o - 1, o, o + 1, o + 2)}
        cand = [(xx, y) for y in range(grid.y) for xx in range(grid.x) if free(xx, y)]
        if cfg.dn_cores == "near":
            cand.sort(key=lambda c: c[0] not in near_cols)
        else:
            cand.sort(key=lambda c: c[0] in near_cols)
        d = cand[: cfg.dn_n]
        assert len(d) == cfg.dn_n
    return grid, opt, w, x, d


def _x_reads(cfg, perm_rows, rowb):
    """All x reads as (bank, offset) in issue order (expert-major, row, segment)."""
    if cfg.x_layout == "dense":
        per_bank = -(-len(perm_rows) // NB) * rowb
        return [(b, k * cfg.x_sz) for k in range(per_bank // cfg.x_sz) for b in range(NB)]
    segs = rowb // cfg.x_sz
    assert segs * cfg.x_sz == rowb, (rowb, cfg.x_sz)
    return [(r % NB, (r // NB) * rowb + s * cfg.x_sz) for r in perm_rows for s in range(segs)]


def _assign_x(cfg, reads, xcores):
    lists = [[] for _ in xcores]
    if cfg.x_assign == "pin":
        by_bank = {}
        for i, (_, b) in enumerate(xcores):
            by_bank.setdefault(b, []).append(i)
        cnt = [0] * NB
        for b, off in reads:
            ks = by_bank[b]
            lists[ks[cnt[b] % len(ks)]].append((b, off))
            cnt[b] += 1
    else:
        n = len(xcores)
        per = -(-len(reads) // n)
        for i in range(n):
            lists[i] = reads[i * per : (i + 1) * per]
    return lists


def _w_plan(pat, sz, wcores, w_bytes):
    """(g0, n, gstride) a weight core: pin = its bank's chunks in order, rot = consecutive chunks (rotating banks)."""
    chunks_pb = w_bytes // NB // sz
    if pat == "pin":
        by_bank = {}
        for i, (_, b) in enumerate(wcores):
            by_bank.setdefault(b, []).append(i)
        out = [None] * len(wcores)
        for b, ks in by_bank.items():
            n = chunks_pb // len(ks)
            for s, i in enumerate(ks):
                out[i] = (b + NB * s * n, n, NB)
        return out
    tot = chunks_pb * NB
    n = tot // len(wcores)
    return [(i * n, n, 1) for i in range(len(wcores))]


class _Bufs:
    def __init__(self, device, model):
        self.model = model
        w_bytes, xrows, rowb = _vol(model)
        self.w_bytes, self.xrows, self.rowb = w_bytes, xrows, rowb
        CH = 65536
        w_pages = -(-w_bytes // CH // NB) * NB + NB  # (+1 page a bank of slack)
        self.w = ttnn.allocate_tensor_on_device(
            ttnn.Shape([w_pages, CH // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        self.d_bytes = w_bytes // 3  # (down = a third of the weights)
        self.d = ttnn.allocate_tensor_on_device(
            ttnn.Shape([-(-self.d_bytes // CH // NB) * NB + NB, CH // 4]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        x_pages = -(-xrows // NB) * NB
        self.x = ttnn.allocate_tensor_on_device(
            ttnn.Shape([x_pages, rowb // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        g = torch.Generator().manual_seed(1234)
        self.perm = torch.randperm(xrows, generator=g).tolist()  # experts = consecutive M-row slices


_BUFS = {}


def _bufs(device, model):
    if model not in _BUFS:
        _BUFS.clear()
        _BUFS[model] = _Bufs(device, model)
    return _BUFS[model]


def _program(device, cfg, bufs, mode, wcores, xcores, dcores):
    """mode: 'w' / 'x' / 'wx'. Returns (program, results tensor, table tensor, kernel entries [(role, k)], bytes)."""
    use_w, use_x = "w" in mode, "x" in mode
    gu_bytes = bufs.w_bytes - bufs.d_bytes if cfg.dn_n else bufs.w_bytes
    wplan = _w_plan(cfg.w_pat, cfg.w_sz, wcores, gu_bytes)
    xl = _assign_x(cfg, _x_reads(cfg, bufs.perm, bufs.rowb), xcores) if use_x else [[] for _ in xcores]
    gap = int(cfg.x_pace_us * CLK_MHZ / max(len(l) for l in xl)) if (cfg.x_pace_us and use_x) else 0
    W = dict(wsz=cfg.w_sz, wbuf=bufs.w)
    groups = []  # dict(role, cores, per [(nw, g0, gstride, x list)], depth, noc, risc, wsz, wbuf)
    if cfg.x_mode == "wq":
        per = [((p[1] if use_w else 0), p[0], p[2], l) for p, l in zip(wplan, xl)]
        groups.append(dict(role="wx", cores=[c for c, _ in wcores], per=per, depth=cfg.w_q, noc=cfg.w_noc, risc=0, **W))
    else:
        if use_w:
            per = [(p[1], p[0], p[2], []) for p in wplan]
            groups.append(
                dict(role="w", cores=[c for c, _ in wcores], per=per, depth=cfg.w_q, noc=cfg.w_noc, risc=0, **W)
            )
        if use_x:
            risc = 1 if cfg.x_mode == "ncrisc" else 0
            noc = 1 if cfg.x_mode == "ncrisc" else cfg.x_noc
            per = [(0, 0, 1, l) for l in xl]
            groups.append(dict(role="x", cores=[c for c, _ in xcores], per=per, depth=cfg.x_q, noc=noc, risc=risc, **W))
    if use_w and cfg.dn_n:
        dplan = _w_plan("rot", cfg.dn_sz, [(c, None) for c in dcores], bufs.d_bytes)
        per = [(p[1], p[0], p[2], []) for p in dplan]
        groups.append(dict(role="d", cores=dcores, per=per, depth=cfg.dn_q, noc=0, risc=0, wsz=cfg.dn_sz, wbuf=bufs.d))
    nk = sum(len(g["cores"]) for g in groups)
    tabw = -(-max([len(l) for l in xl] + [1]) // 16) * 16
    tab = torch.zeros(nk, tabw, dtype=torch.int64)
    res = ttnn.from_torch(
        torch.zeros(nk, 16, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k, total, entries = 0, 0, []
    for g in groups:
        g["idx"] = []
        for nw, g0, gs, xs in g["per"]:
            for j, (b, off) in enumerate(xs):
                tab[k, j] = (b << 28) | off
            g["idx"].append(k)
            entries.append((g["role"], k))
            total += nw * g["wsz"] + len(xs) * cfg.x_sz
            k += 1
    tab_t = ttnn.from_torch(
        tab.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tab_bytes = tabw * 4
    allcores = sorted({c for g in groups for c in g["cores"]})
    crs_all = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in allcores])
    kernels, cbs = [], []
    for cbi, g in enumerate(groups):
        slot = max(g["wsz"] if g["role"] != "x" else 0, cfg.x_sz if g["role"] in ("x", "wx") else 0)
        rt = ttnn.RuntimeArgs()  # (filled before the KernelDescriptor: it copies them)
        for c, (nw, g0, gs, xs), idx in zip(g["cores"], g["per"], g["idx"]):
            rt[c[0]][c[1]] = [
                int(g["wbuf"].buffer_address()),
                int(bufs.x.buffer_address()),
                int(tab_t.buffer_address()),
                int(res.buffer_address()),
                idx,
                nw,
                g0,
                gs,
                len(xs),
                gap,
            ]
        size = g["depth"] * slot + tab_bytes + 64
        cbs.append(
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=crs_all,
                format_descriptors=[ttnn.CBFormatDescriptor(cbi, ttnn.uint32, size)],
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_K,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in g["cores"]]
                ),
                compile_time_args=[
                    g["depth"],
                    g["wsz"],
                    cfg.x_sz,
                    slot,
                    cbi,
                    cfg.order if g["role"] == "wx" else 0,
                    tab_bytes,
                    int(cfg.x_batch and g["role"] == "x"),
                ],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1 if g["risc"] else ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.NOC_1 if g["noc"] else ttnn.NOC.NOC_0,
                ),
            )
        )
    prog = ttnn.ProgramDescriptor()
    prog.cbs, prog.kernels = cbs, kernels
    prog.custom_program_hash = zlib.crc32(f"dxw {asdict(cfg)} {mode} {tabw}".encode())
    return prog, res, tab_t, entries, total


def _spans(res, entries):
    v = ttnn.to_torch(res).to(torch.int64) & 0xFFFFFFFF
    ref = int(v[0, 0])

    def rel(t):
        d = (int(t) - ref) & 0xFFFFFFFF
        return d - (1 << 32) if d >= (1 << 31) else d

    out = {}
    for role, k in entries:
        t0, t1 = rel(v[k, 0]), rel(v[k, 1])
        a = out.setdefault(role, [t0, t1, 0])
        a[0], a[1] = min(a[0], t0), max(a[1], t1)
    tot = [min(a[0] for a in out.values()), max(a[1] for a in out.values())]
    us = {r: (a[1] - a[0]) / CLK_MHZ for r, a in out.items()}
    us["all"] = (tot[1] - tot[0]) / CLK_MHZ
    return us


def _run_mode(device, cfg, bufs, mode, wcores, xcores, dcores):
    prog, res, tab_t, entries, total = _program(device, cfg, bufs, mode, wcores, xcores, dcores)
    samples = []
    with _Clk() as clk:
        for _ in range(REPS + 1):  # (first run compiles / warms)
            ttnn.generic_op([bufs.w, bufs.d, bufs.x, tab_t, res], prog)
            ttnn.synchronize_device(device)
            samples.append(_spans(res, entries))
    samples = samples[1:]
    med = {r: statistics.median(s[r] for s in samples) for r in samples[0]}
    # (cross-check of the 1350 MHz wall-clock conversion: host time a launch over 10 back-to-back launches,
    # = the device span + the dispatch gap between programs)
    t = time.perf_counter()
    for _ in range(10):
        ttnn.generic_op([bufs.w, bufs.d, bufs.x, tab_t, res], prog)
    ttnn.synchronize_device(device)
    med["host"] = (time.perf_counter() - t) / 10 * 1e6
    logger.info(
        "DXW aiclk per card " + " ".join(f"{k.split('!')[-1].split('/')[0]}:{v[0]}-{v[1]}" for k, v in clk.per.items())
    )
    ttnn.deallocate(res)
    ttnn.deallocate(tab_t)
    return med, total, (min(clk.seen, default=0), max(clk.seen, default=0))


def _pct(b, us):
    return b / us / 1e3 / 512 * 100


@pytest.mark.parametrize("name", list(V))
def test_dram_xw_mix(device, name):
    cfg = V[name]
    bufs = _bufs(device, cfg.model)
    grid, opt, wcores, xcores, dcores = _placement(device, cfg)
    logger.info(
        f"DXW {name} grid {grid.x}x{grid.y} w cores {[c for c, _ in wcores]} x cores {[c for c, _ in xcores]}"
        f" down cores {dcores}"
    )
    row = {"name": name, "cfg": asdict(cfg)}
    for mode in ("w", "x", "wx"):
        med, total, clk = _run_mode(device, cfg, bufs, mode, wcores, xcores, dcores)
        row[mode] = {"us": med, "bytes": total, "pct": _pct(total, med["all"]), "aiclk": clk}
        logger.info(
            f"DXW {name:12s} {mode:2s} {total / 1e6:7.1f} MB in {med['all']:7.1f} us = {total / med['all'] / 1e3:6.1f} GB/s"
            f" ({_pct(total, med['all']):5.1f}%)  roles {', '.join(f'{r} {u:.1f}' for r, u in med.items() if r != 'all')}"
            f"  aiclk {clk[0]}-{clk[1]}"
        )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info(
        f"DXWSUM {name:12s} w {row['w']['us']['all']:6.1f} us ({row['w']['pct']:4.1f}%) | x {row['x']['us']['all']:6.1f}"
        f" us ({row['x']['pct']:4.1f}%) | both {row['wx']['us']['all']:6.1f} us ({row['wx']['pct']:4.1f}%)"
    )
