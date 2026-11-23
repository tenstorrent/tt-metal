# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Bank-streamed decode linear, bit-equal to tt-metal's multi-core DRAM-sharded matmul.

``y = x @ W`` for one token. The reference is ``ttnn.linear`` with a width-sharded DRAM weight (``Nbt`` tiles per
bank) and ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=KB, num_workers_per_dram_bank=C,
use_multicore_pipeline=True)``, fp32 Dest. That op gives each core of bank b a column group cn (CN = min(C, Nbt)
groups per bank), splits the group's columns into P passes of <= 8 tiles, keeps each pass's whole K in fp32 Dest and
walks the K blocks starting at block rot = grp * nblk / NGRP (grp = b * CN + cn, NGRP = banks * CN), wrapping
(bmm_dram_sharded_multicore.cpp). Each output tile depends only on that order, so this op replays it exactly:

* layout: per bank, the tiles of every pass stored in the order compute eats them ([K block in walk order][row r
  of the block][column of the pass]), the passes of one column group back to back, so a reader core streams one
  contiguous byte range of its own bank;
* compute: per pass, matmul_block(x[j * KB + r], w, ..., ct_dim = ncq) over the same walk, fp32 Dest, same fidelity;
* placement: one core per (bank, column group) on the bank's optimal NOC0 reader row (moe_bank.row_placement).

Several reference ops with the same shape and config (e.g. the 14 unembed column chunks) become one program: each
core runs its column group of every chunk. Output: row 0 of every output tile into a dense fp32 row (value n at byte
4 n), the layout tt/fused/head.py's FusedLogitsTail reads.
"""

import math
import os

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.moe_bank import row_placement

TILE = 32
DST_TILES = 8  # fp32 Dest tiles per pass (full sync), as the reference
# HALF: when only the first x_rows <= 16 rows of x are in use and the output is whole tiles, x and the output are
# 16 x 32 tiles: the matmul runs half the FPU passes and the x fetch moves half the bytes. Rows 0..15 of every output
# tile come out the same (the first half of each 32 x 32 output page); rows 16..31 are not written. Attention w_all
# (x_rows 9), traced decode: 4K 15.08 -> 14.89 ms, 128K 20.41 -> 20.25, bit-equal.
HALF = os.environ.get("INKLING_LINEAR_HALF", "1") == "1"


def half_tiles(x_rows, tiles_out, out_dtype):
    """-> (half, x / output TileDescriptor, x tile bytes, output tile bytes) of a call."""
    half = HALF and tiles_out and 0 < x_rows <= 16
    t = ttnn.Tile([16 if half else TILE, TILE])
    return half, ttnn.TileDescriptor(t), t.get_tile_size(ttnn.bfloat16), t.get_tile_size(out_dtype)


def reference_schedule(Kt, Nt, Nbt, C, kb):
    """The reference op's work split: [(bank, cn, rot, [(first column, ncq) per pass])], KB, nblk."""
    CN = min(C, Nbt)
    assert C <= Nbt, "K row groups (C > Nbt) are not replayed"
    KB = min(kb, Kt)
    assert Kt % KB == 0, (Kt, KB)
    nblk = Kt // KB
    P = math.ceil(math.ceil(Nbt / CN) / DST_TILES)
    banks = math.ceil(Nt / Nbt)
    out = []
    for b in range(banks):
        for cn in range(CN):
            n0, n1 = Nbt * cn // CN, Nbt * (cn + 1) // CN
            nc = n1 - n0
            rot = (b * CN + cn) * nblk // (banks * CN)
            passes = [(b * Nbt + n0 + nc * q // P, nc * (q + 1) // P - nc * q // P) for q in range(P)]
            out.append((b, cn, rot, passes))
    return out, KB, nblk


def _alt_block(p, n_alt, nblk):
    """Block p of a pass is one of the n_alt blocks stored in the alt bank (evenly spaced; bank_linear_reader.cpp)."""
    return (p + 1) * n_alt // nblk > p * n_alt // nblk


def _spread(items, nb, ktiles, kb):
    """[(alt bank, blocks)] per item: whole first-pass blocks of the over-full banks' largest items move to the
    under-full banks (vs the mean) while that lowers the larger of the two banks' tile counts."""
    size = [sum(ncq for ncq, _ in part) * ktiles for _, _, part in items]
    load = [0] * nb
    for (b, _, _), n in zip(items, size):
        load[b] += n
    mean = sum(load) / nb
    takers = [c for c in range(nb) if load[c] < mean]
    alt = [(b, 0) for b, _, _ in items]
    for i in sorted(range(len(items)), key=lambda i: -size[i]):
        b, _, part = items[i]
        if load[b] <= mean or not takers:
            continue
        blk = kb * part[0][0]
        lo = min(takers, key=lambda c: load[c])
        n = 0
        while n < ktiles // kb and load[b] - blk >= max(mean, load[lo] + blk):
            load[b] -= blk
            load[lo] += blk
            n += 1
        if n:
            alt[i] = (lo, n)
    return alt


# noc0_flow_links / plan_noc0_workers: adapted from Lottolabs' Inkling-2-Mini P150 runtime
# (runtime/inkling2/tt/moe.py, https://huggingface.co/Lottolabs/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150),
# Copyright Lottolabs, Licensed under the Apache License, Version 2.0.
# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net> (changes: per-bank counts, p100a use)
def noc0_flow_links(anchor, core, half, rows, south=10):
    """NOC0 links of the DRAM read responses from the bank at `anchor`'s row to `core` (logical coordinates): east
    along the endpoint row, then south (wrapping). -> (links, hops); `south` weights a south hop against an east one."""
    ax, ay = anchor
    x, y = core
    links = {("E", ay, c) for c in range(half[0], x + 1)}
    dy = (y - ay) % rows
    links |= {("S", x, (ay + i + 1) % rows) for i in range(dy)}
    wrap = 2 if dy and y < ay else 0
    return links, (x - half[0] + 1) + south * (dy + wrap)


def noc0_placement(device, per_bank, south=10):
    """per_bank[b] logical cores for DRAM bank b whose NOC0 response paths share as few links with other banks' as
    possible: dealt one per bank per round, each the free core of the bank's half (its anchor column to the next
    anchor column) with the fewest links used by other banks, then the fewest hops."""
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    anchors = [(int(c.x), int(c.y)) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    starts = sorted({a[0] for a in anchors})
    half_of = {s_: (s_, ([t for t in starts if t > s_] or [gx])[0]) for s_ in starts}
    free = {(x, y) for x in range(gx) for y in range(gy)}
    used, out = {}, {b: [] for b in range(len(anchors))}
    for rnd in range(max(per_bank)):
        for b, anchor in enumerate(anchors):
            if rnd >= per_bank[b]:
                continue
            half = half_of[anchor[0]]
            best = None
            for c in sorted(free):
                if not half[0] <= c[0] < half[1]:
                    continue
                links, hops = noc0_flow_links(anchor, c, half, gy, south)
                key = (sum(1 for k in links if used.get(k, b) != b), hops, c[1], c[0])
                if best is None or key < best[0]:
                    best = (key, c, links)
            assert best is not None, f"no free worker left for bank {b}"
            free.discard(best[1])
            for k in best[2]:
                used.setdefault(k, b)
            out[b].append(best[1])
    return out


PLACEMENT = "row"  # BankLinear worker placement: "row" (moe_bank.row_placement) or "noc0" (noc0_placement)


class BankLinear:
    def __init__(
        self, device, w, dtype, chunk, kb, cores_per_bank, compute_kernel_config, out_cols=None, out_dtype=ttnn.float32,
        split=1, pass_cols=None, spread=False, placement=None,
    ):  # fmt: skip
        """w: torch [K, N]; ``chunk`` columns per reference op (a multiple of banks * 32; N is zero-padded to whole
        chunks, as tt/dram_sharded.DramShardedLinear); kb / cores_per_bank: the reference in0_block_w /
        num_workers_per_dram_bank. out_cols: columns written (default N). out_dtype: the reference's output dtype
        (fp32 or bf16; the tile is packed from fp32 Dest as the reference packs it). split: cores per reference
        column group (its passes are divided between them; every pass is its own Dest chain, so the output is the
        same). pass_cols: cut passes to at most this many columns (fewer columns per matmul_block call).
        spread: even out the banks' bytes: evenly spaced blocks of the busiest items of over-full banks are stored in
        (and read from) under-full banks (same tiles, same order into compute, so the output is the same)."""
        nb = device.dram_grid_size().x
        K, N = w.shape
        n_chunks = math.ceil(N / chunk)
        w = torch.nn.functional.pad(w.bfloat16(), (0, n_chunks * chunk - N))  # BFP values are exact in bf16
        self.K, self.N, self.Kt = K, N, K // TILE
        self.chunk, self.n_chunks = chunk, n_chunks
        self.out_cols = out_cols or N
        Nbt = chunk // TILE // nb
        sched, self.KB, self.nblk = reference_schedule(self.Kt, chunk // TILE, Nbt, cores_per_bank, kb)
        tiles = w.reshape(self.Kt, TILE, -1, TILE).permute(0, 2, 1, 3)  # [Kt, Nt, 32, 32]
        # core work items: (bank, rot, [(ncq, first output column tile)]) - one reference group split over `split` cores
        self.items = []
        for b, cn, rot, passes in sched:
            todo = []
            for c in range(n_chunks):
                for first, ncq in passes:
                    g0 = c * (chunk // TILE) + first
                    ncq = min(ncq, max(0, -(-self.out_cols // TILE) - g0))  # columns past out_cols: not computed
                    if ncq:
                        step = pass_cols or ncq
                        todo += [(min(step, ncq - i), g0 + i) for i in range(0, ncq, step)]
            for s_ in range(split):
                part = todo[len(todo) * s_ // split : len(todo) * (s_ + 1) // split]
                if part:
                    self.items.append((b, rot, part))
        # spread: alt[i][1] blocks of item i's first pass, spaced evenly (_alt_block), live in bank alt[i][0]
        self.alt = [(b, 0) for b, _, _ in self.items]
        if spread:
            self.alt = _spread(self.items, nb, self.KB * self.nblk, self.KB)
        per_bank = [[] for _ in range(nb)]
        self.offsets, self.alt_offsets = [], []
        for (b, rot, part), (ab, na) in zip(self.items, self.alt):
            self.alt_offsets.append(sum(t.shape[0] for t in per_bank[ab]))
            on_alt = lambda q, p: q == 0 and _alt_block(p, na, self.nblk)  # noqa: E731
            ncq, g0 = part[0]
            for p in range(self.nblk):
                if on_alt(0, p):
                    j = (p + rot) % self.nblk
                    for r in range(self.KB):
                        per_bank[ab].append(tiles[j * self.KB + r, g0 : g0 + ncq])
            self.offsets.append(sum(t.shape[0] for t in per_bank[b]))  # first tile slot of this item in bank b
            for q, (ncq, g0) in enumerate(part):
                for p in range(self.nblk):
                    if on_alt(q, p):
                        continue
                    j = (p + rot) % self.nblk
                    for r in range(self.KB):
                        per_bank[b].append(tiles[j * self.KB + r, g0 : g0 + ncq])
        stacks = [torch.cat([t.reshape(-1, TILE, TILE) for t in tb]) if tb else None for tb in per_bank]
        rows = max(len(s_) for s_ in stacks if s_ is not None)
        pages = torch.zeros(rows, nb, TILE, TILE, dtype=torch.bfloat16)
        for b, s_ in enumerate(stacks):
            if s_ is not None:
                pages[: len(s_), b] = s_
        del stacks, per_bank, tiles, w
        # page p = slot * nb + bank (DRAM interleaved)
        self.weight = ttnn.from_torch(pages.reshape(1, 1, -1, TILE), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        self.device, self.nb, self.dtype = device, nb, dtype
        counts = [sum(1 for it in self.items if it[0] == b) for b in range(nb)]
        self.placement = placement or PLACEMENT
        place = noc0_placement(device, counts) if self.placement == "noc0" else row_placement(device, counts)
        seen = [0] * nb
        self.cores = []
        for i, (b, rot, part) in enumerate(self.items):
            self.cores.append((place[b][seen[b]], i))
            seen[b] += 1
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c, _ in self.cores])
        self.max_ncq = max(ncq for _, _, part in self.items for ncq, _ in part)
        self.out_dtype = out_dtype
        # x_rows > 0: the caller guarantees only the first x_rows rows of every x are non-zero (x_rows.hpp)
        self.x_rows = 0
        # tile output memory (decode consumers that read it right away may take it in L1); out_buf: a persistent
        # output tensor reused by every call (its consumers must be done with it before the next call)
        self.out_mem = ttnn.DRAM_MEMORY_CONFIG
        self.out_buf = None
        cfg = compute_kernel_config
        self.compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=cfg.math_fidelity, fp32_dest_acc_en=True, dst_full_sync_en=True,
            math_approx_mode=cfg.math_approx_mode,
        )  # fmt: skip
        assert cfg.fp32_dest_acc_en, "the replayed schedule is the fp32 Dest one"

    def reader_args(self, i):
        """bank_linear_reader.cpp runtime args of work item i (+ reader_defines())."""
        b, rot, part = self.items[i]
        args = [self.weight.buffer_address(), b, self.offsets[i], len(part)] + [n for n, _ in part]
        return args + ([*self.alt[i], self.alt_offsets[i]] if self.spreads() else [])

    def spreads(self):
        return any(n for _, n in self.alt)

    def reader_defines(self):
        return [("W_SPREAD", "1")] * self.spreads()

    def __call__(self, x, out=None, head=None):
        """x [1,1,M,K] bf16 TILE, M <= 32 (any memory). out=None: returns [1,1,M,out_cols] TILE DRAM interleaved
        in out_dtype (the reference's output after to_memory_config(DRAM) + slice: every row of each tile). out given:
        a dense fp32 ROW_MAJOR [1,1,1,W] row (W >= out_cols) receiving row 0 only (tt/fused/head.py's raw row), or a
        list of M such rows (R verify rows: row r of x -> out[r]); the rows may also be [1,1,1,W] fp32 TILE tensors
        (row 0 of each tile written; the pages spread over the DRAM banks).
        head (with out rows): the LM head in this program: dict(inv, cap: fp32 bits of 1 / softcap and softcap,
        n_arg: argmax columns, tokens: one uint32 [1] DRAM tensor per row): out rows get tanh(x W * inv) * cap and
        tokens[r] the argmax of row r (bank_linear_compute / _writer: the logits tail's math and tie rule)."""
        outs = out if isinstance(out, (list, tuple)) else [out]
        out = outs[0]
        tb = ttnn.Tile([TILE, TILE]).get_tile_size(self.dtype)
        tiles_out = out is None
        if tiles_out and self.out_buf is not None:
            out = self.out_buf
            assert tuple(out.shape) == (1, 1, x.shape[-2], self.out_cols), (out.shape, x.shape)
        elif tiles_out:
            shape = ttnn.Shape([1, 1, x.shape[-2], self.out_cols])
            out = ttnn.allocate_tensor_on_device(shape, self.out_dtype, ttnn.TILE_LAYOUT, self.device,
                                                 self.out_mem)  # fmt: skip
        else:
            assert self.out_dtype == ttnn.float32 and x.shape[-2] == len(outs)
        otb = ttnn.Tile([TILE, TILE]).get_tile_size(self.out_dtype)
        half, th, xtb, htb = half_tiles(self.x_rows, tiles_out, self.out_dtype)
        row_tiles = not tiles_out and out.layout == ttnn.TILE_LAYOUT  # out rows as [1, 1, 1, W] TILE tensors
        named = dict(kt=self.Kt, kb=self.KB, nblk=self.nblk, n_banks=self.nb, out_cols=self.out_cols,
                     out_page=otb if (tiles_out or row_tiles) else out.shape[-1] * 4, max_ncq=self.max_ncq, tiles_out=int(tiles_out),
                     out_tile=htb, x_rows=self.x_rows, raw_rows=1 if tiles_out else len(outs), x_half=int(half))  # fmt: skip
        r_args, c_args, w_args = {}, {}, {}
        if head is not None:
            assert not tiles_out and len(head["tokens"]) == len(outs)
            # the merging worker: the one with the fewest output tiles (it waits for the others at the end)
            load = [sum(n for n, _ in self.items[i][2]) for _, i in self.cores]
            mi = min(range(len(load)), key=lambda k: load[k])
            m = self.device.worker_core_from_logical_core(ttnn.CoreCoord(*self.cores[mi][0]))
        for k, (c, i) in enumerate(self.cores):
            slot = (k - mi) % len(self.cores) if head is not None else k  # the merging worker is slot 0
            b, rot, part = self.items[i]
            core = ttnn.CoreCoord(*c)
            r_args[core] = self.reader_args(i)
            c_args[core] = [rot, len(part)] + [n for n, _ in part]
            w_args[core] = [x.buffer_address(), rot, out.buffer_address(), len(part)] + [v for p_ in part for v in p_]
            if head is not None:
                w_args[core] += [slot, m.x, m.y, len(self.cores)]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        blk = self.KB * self.max_ncq
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_reader.cpp", core_ranges=self.grid,
                compile_time_args=acc(self.weight), named_compile_time_args=list(named.items()),
                defines=self.reader_defines(), runtime_args=[(k, v) for k, v in r_args.items()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_compute.cpp", core_ranges=self.grid,
                named_compile_time_args=list(named.items()), runtime_args=[(k, v) for k, v in c_args.items()],
                defines=[("BL_SOFTCAP", str(head["inv"])), ("BL_CAP", str(head["cap"])),
                         ("BL_SFPU_IT", str(2 * -(-x.shape[-2] // 4)))] if head else [],  # fmt: skip
                config=self.compute,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_writer.cpp", core_ranges=self.grid,
                compile_time_args=acc(x) + acc(out), named_compile_time_args=list(named.items()),
                defines=([("BL_HEAD", str(head["n_arg"]))] if head else []) + ([("BL_ROW_TILES", "1")] if row_tiles else []),
                runtime_args=[(k, v) for k, v in w_args.items()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]  # fmt: skip
        t32 = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))

        def cb(i, fmt, page, n, tile=t32):
            return ttnn.CBDescriptor(total_size=page * n, core_ranges=self.grid,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page, tile=tile)])  # fmt: skip

        cbs = [
            cb(0, ttnn.bfloat16, xtb, self.Kt, th),  # x, whole K resident
            cb(1, self.dtype, tb, 4 * blk),  # weight blocks: 4 fixed slots (reader keeps 3 in flight)
            cb(16, self.out_dtype, htb, 2 * DST_TILES, th),  # output tiles of a pass
        ]
        common = [o.buffer_address() for o in outs[1:]]
        sems, extra = [], []
        if head is not None:
            common += [t.buffer_address() for t in head["tokens"]]
            n = len(self.cores)
            cbs.append(cb(17, ttnn.float32, 16 * len(outs) * (n + 1), 1))  # argmax slots (bank_linear_writer)
            sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.grid, initial_value=0)]
            extra = list(head["tokens"])
        if common:
            kernels[2].common_runtime_args = common
        prog = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)
        return ttnn.generic_op([x, self.weight, *outs[1:], *extra, out], prog)
