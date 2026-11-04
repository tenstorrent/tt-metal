# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill megakernel (PMK) cache-write phase + SEGMENT TABLE (doc/ttnn/16-prefill-production/prefill_megakernel.md
§0 "cache": P9 and the MLA latent pages / indexer keys; §1 segments; §4b hand-off rule).

Inside the persistent layer program, on reader / writer cores (never owners):
  latent_parts  sparse_dsa: the layer's latent rows [T, 512] bf16 -> each segment's slot pages of the paged latent cache
                (whole 64-row pages; the NULL-block rule for pad pages)              kernels/pmk/cache_lat.cpp
  dsa_parts     sparse_dsa: pooled keys [T / 4, 128] -> the slot's pooled-key store pages; the slot's ring rows from the
                [k | g] rows                                                         kernels/pmk/cache_dsa.cpp
  state_parts   P9 (KDA layers): every segment's final S [H_r, 128, 128] fp32 + conv carry [3, W_r] -> its slot of the
                decode stack's state store (StateIO's bytes and layout)              kernels/pmk/cache_state.cpp
Every part waits on the producer's monotonic block counter with noc_semaphore_wait_min (>= k, never equality: §4b) --
WAIT_BLOCKS per 32-row source block (the latent / kg rows land block by block under the next phase's compute) or
WAIT_ALL once (the scan chains' final-state counter) -- and runs standalone with WAIT_NONE (the device unit).

The SEGMENT TABLE (SegTable; kernels/pmk/cache_common.hpp is the kernel-side twin) is one uint32 ROW_MAJOR page the
host writes before every replay: header + NSEG_MAX segments (slot, row0, n_valid, pos0, page_row, flags, n_rows). v1
= one segment at row 0; v2 = K packed / continuation segments on 128-row boundaries. It lives in DRAM (default) or in
an L1 CB region beside the stack's ARG CB 30 (SEGTAB_CB / SEGTAB_OFF defines: no read). The plan_* functions are the
kernels' index math in Python (the layout tests and the device unit's expected bytes)."""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "pmk")  # absolute: a worktree's kernels
TILE = 32
BLOCK = 64  # latent page rows (prefill_tp.py BLOCK)
KV_LORA = 512
KX = KV_LORA // TILE  # 16 tiles per latent row-tile
TPB = (BLOCK // TILE) * KX  # 32 tiles per latent page (64 KiB)
KPOOL = 4
POOL_ROWS = 128  # tokens per pooled-key store page (32 pools x 4)
INDEX_DIM = 128
POOL_TILES = INDEX_DIM // TILE  # 4 tiles per store page (8 KiB)
KG_W = 2 * INDEX_DIM  # a [k | g] row
RING_TILES = KG_W // TILE  # 8 tiles per slot
SEG_ALIGN = 128  # segments start on 128-row boundaries (prefill_tp.py packed_align with the DSA indexer)
NSEG_MAX = 8  # v2: segments per replay (fixed per trace key)
NULL_BLOCK = 0  # vLLM's null block (prefill_tp.py NULL_BLOCK)
SEG_NONE = 0xFFFFFFFF
SEG_MAGIC = 0x31474553  # "SEG1"
HDR_WORDS, SEG_WORDS = 8, 8
F_SAMPLE, F_FRESH, F_CONT = 1, 2, 4
WAIT_NONE, WAIT_ALL, WAIT_BLOCKS = 0, 1, 2
TILE_BF16, TILE_FP32 = 2048, 4096
# Semaphore ids (prefill_megakernel.md §4b-ids: 16 per core, 0..15 all assigned on the shared cores). The writers run
# on the 4 TAIL cores (idle between P8 and P12; they draw 12 / 13 there and never the MM / member / KDA roles), so on
# the WRITER cores ids 4 / 5 / 6 are free and are the producers' block counters raised on the writer cores -- never
# on an MM core, a member or a lane. No write-done counter: the program boundary orders the writes before the next
# replay reads the pages. If the family places writers on MM cores instead, it must pass other ids (open item).
SEM_LAT = 4  # latent rows landed, +1 per 32-row block (the MLA latent producer -> cache_lat; HN's id, not its core)
SEM_KG = 5  # [k | g] rows + pooled keys landed, +1 per 32-row block (the indexer key producer -> cache_dsa)
SEM_STATE = 6  # KDA scan chains' final states landed, +1 per chain (P6 -> P9)
CB_SCR = 37  # the writers' scratch CB: free in both halves of §4b's map (owners 0..31 + 32..63 KDA; readers 36, 38..)
BF16, FP32 = ttnn.bfloat16, ttnn.float32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _a64(b: int) -> int:
    return (b + 63) // 64 * 64


def table_bytes(nseg_max: int = NSEG_MAX) -> int:
    return 4 * (HDR_WORDS + SEG_WORDS * nseg_max)


# ---------------------------------------------------------------------------------------------------- segment table
@dataclasses.dataclass
class Segment:
    """one request's rows in the packed T rows: rows [row0, row0 + n_valid) carry its tokens at positions pos0 + i"""

    slot: int
    n_valid: int
    row0: int = 0
    pos0: int = 0  # chunk start (0: a fresh request)
    page_row: Optional[int] = None  # the slot's row of the page table (default: the slot)
    sample: bool = True  # the head samples the last valid row
    cont: bool = False  # attends the slot's cached prefix (pos0 > 0)

    @property
    def flags(self) -> int:
        return (F_SAMPLE if self.sample else 0) | (F_FRESH if self.pos0 == 0 else 0) | (F_CONT if self.cont else 0)


class SegTable:
    """the device-side segment table of one replay: T packed rows, segments on SEG_ALIGN boundaries (back to back
    unless row0 is given), the rest padding"""

    def __init__(self, t_rows: int, segments: list, nseg_max: int = NSEG_MAX, align: int = SEG_ALIGN):
        assert t_rows % align == 0 and 0 < len(segments) <= nseg_max, (t_rows, len(segments), nseg_max)
        self.t_rows, self.nseg_max, self.align = int(t_rows), int(nseg_max), int(align)
        self.segments = []
        o = 0
        for s in segments:
            s = dataclasses.replace(s)
            if s.row0 is None or (s.row0 == 0 and o > 0):
                s.row0 = o
            assert s.row0 >= o and s.row0 % align == 0, (s, o)
            assert s.n_valid > 0 and s.pos0 % max(BLOCK, align) == 0, s
            if s.page_row is None:
                s.page_row = s.slot
            n_rows = -(-s.n_valid // align) * align
            assert s.row0 + n_rows <= t_rows, f"segment {s} past T={t_rows}"
            o = s.row0 + n_rows
            self.segments.append((s, n_rows))
        assert len({s.slot for s, _ in self.segments}) == len(self.segments), "distinct slots"

    def words(self) -> torch.Tensor:
        w = torch.zeros(HDR_WORDS + SEG_WORDS * self.nseg_max, dtype=torch.int64)
        w[:7] = torch.tensor([SEG_MAGIC, len(self.segments), self.t_rows, self.align, BLOCK, POOL_ROWS, 0])
        for k in range(self.nseg_max):
            base = HDR_WORDS + SEG_WORDS * k
            if k < len(self.segments):
                s, n_rows = self.segments[k]
                w[base : base + 7] = torch.tensor([s.slot, s.row0, s.n_valid, s.pos0, s.page_row, s.flags, n_rows])
            else:
                w[base] = SEG_NONE
        return w

    def tensor(self, mesh, mapper=None):
        """the persistent device tensor [1, words] uint32 ROW_MAJOR (one page); write() refreshes it per replay"""
        return ttnn.from_torch(self.words().reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=mesh, memory_config=DRAM, mesh_mapper=mapper)  # fmt: skip

    def write(self, dst, mapper=None):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(self.words().reshape(1, -1), dtype=ttnn.uint32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper), dst)  # fmt: skip

    # the kernel accessor's twin
    def seg_of_row(self, r: int):
        for k, (s, n_rows) in enumerate(self.segments):
            if s.row0 <= r < s.row0 + n_rows:
                return k
        return None

    def valid(self, k: int, r: int) -> bool:
        s = self.segments[k][0]
        return 0 <= r - s.row0 < s.n_valid

    def pos(self, k: int, r: int) -> int:
        s = self.segments[k][0]
        return s.pos0 + r - s.row0

    @classmethod
    def decode(cls, words: torch.Tensor) -> "SegTable":
        w = words.reshape(-1).to(torch.int64)
        assert int(w[0]) == SEG_MAGIC
        nseg_max = (w.numel() - HDR_WORDS) // SEG_WORDS
        segs = []
        for k in range(int(w[1])):
            b = HDR_WORDS + SEG_WORDS * k
            slot, row0, nv, pos0, pr, fl = (int(v) for v in w[b : b + 6])
            segs.append(Segment(slot, nv, row0, pos0, pr, bool(fl & F_SAMPLE), bool(fl & F_CONT)))
        return cls(int(w[2]), segs, nseg_max, int(w[3]))


# ---------------------------------------------------------------------------------------------------- the plans
def plan_latent(tab: SegTable, page_table: torch.Tensor, src_row0: int = 0, src_rows: Optional[int] = None,
                write_null: bool = False) -> list:  # fmt: skip
    """[(src page, dst page)] of cache_lat.cpp: src page i = packed rows [64 i, 64 i + 64) (relative to the source's
    rows), dst = the slot's physical page (page_table [rows, blocks_per_seq] int64/int32; -1 = none). Pad pages:
    skipped, or (write_null) -> NULL_BLOCK."""
    src_rows = tab.t_rows if src_rows is None else src_rows
    out = []
    for i in range(tab.t_rows // BLOCK):
        r = i * BLOCK
        if r < src_row0 or r - src_row0 >= src_rows:
            continue
        k = tab.seg_of_row(r)
        if k is None or not tab.valid(k, r):
            if write_null:
                out.append(((r - src_row0) // BLOCK, NULL_BLOCK))
            continue
        s = tab.segments[k][0]
        p = int(page_table[s.page_row][tab.pos(k, r) // BLOCK])
        if p >= 0:
            out.append(((r - src_row0) // BLOCK, p))
    return out


def plan_pools(tab: SegTable, pages_per_slot: int, src_row0: int = 0, src_rows: Optional[int] = None) -> list:
    """[(src store page, dst store page)] of cache_dsa.cpp (a): src page j = packed rows [128 j, 128 j + 128), dst =
    slot x pages_per_slot + pos / 128"""
    src_rows = tab.t_rows if src_rows is None else src_rows
    out = []
    for j in range(tab.t_rows // POOL_ROWS):
        r = j * POOL_ROWS
        if r < src_row0 or r - src_row0 >= src_rows:
            continue
        k = tab.seg_of_row(r)
        if k is None or not tab.valid(k, r):
            continue
        s = tab.segments[k][0]
        out.append(((r - src_row0) // POOL_ROWS, s.slot * pages_per_slot + tab.pos(k, r) // POOL_ROWS))
    return out


def plan_ring(tab: SegTable, src_row0: int = 0, src_rows: Optional[int] = None) -> list:
    """[(slot, [packed row of ring row j or None (a zero row) for j in 0..3])] of cache_dsa.cpp (b) == write_packed's
    ring_rows (prefill_tp.py: rr[j] = o + pos - st if 0 <= pos - st < nv, pos = 4 (end // 4) + j if < end else - 4)"""
    src_rows = tab.t_rows if src_rows is None else src_rows
    out = []
    for s, _ in tab.segments:
        end = s.pos0 + s.n_valid
        base = KPOOL * (end // KPOOL)
        rows = []
        for j in range(KPOOL):
            pos = base + j if base + j < end else base - KPOOL + j
            row = None
            if pos >= 0 and 0 <= pos - s.pos0 < s.n_valid:
                row = s.row0 + pos - s.pos0
                if row < src_row0 or row - src_row0 >= src_rows:
                    row = None
            rows.append(row)
        out.append((s.slot, rows))
    return out


def plan_state(tab: SegTable, chunk: int) -> list:
    """[(segment index, chunk index, row in chunk)] of cache_state.cpp"""
    return [(k, s.slot // chunk, s.slot % chunk) for k, (s, _) in enumerate(tab.segments)]


# the plans applied on host tensors (the device unit's expected bytes; the layout tests)
def apply_latent(plan: list, src: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """src [T, 512] rows, cache [num_blocks, 1, 64, 512] -> the cache after the writes"""
    out = cache.clone()
    for i, p in plan:
        out[p, 0] = src[i * BLOCK : (i + 1) * BLOCK]
    return out


def apply_pools(plan: list, pooled: torch.Tensor, store: torch.Tensor) -> torch.Tensor:
    """pooled [T / 4, 128], store [B x pps, 1, 32, 128]"""
    out = store.clone()
    for j, p in plan:
        out[p, 0] = pooled[j * (POOL_ROWS // KPOOL) : (j + 1) * (POOL_ROWS // KPOOL)]
    return out


def apply_ring(plan: list, kg: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """kg [T, 256], ring [B, 1, 32, 256]"""
    out = ring.clone()
    for slot, rows in plan:
        out[slot, 0] = 0
        for j, r in enumerate(rows):
            if r is not None:
                out[slot, 0, j] = kg[r]
    return out


def apply_state(plan: list, scratches: list, stacks: list) -> list:
    """scratches [(S [1, H, 128, 128], C [1, 3, 1, W])], stacks [(S [chunk, H, 128, 128], C [1, 3, chunk, W])]"""
    out = [(S.clone(), C.clone()) for S, C in stacks]
    for k, cidx, rd in plan:
        out[cidx][0][rd] = scratches[k][0][0]
        out[cidx][1][0, :, rd] = scratches[k][1][0, :, 0]
    return out


# ---------------------------------------------------------------------------------------------------- estimates
def estimate(t_rows: int, tp: int, kind: str, n_seg: int = 1, writers: int = 8, gbps_per_core: float = 10.0) -> dict:
    """bytes written per layer per chip and the write span on `writers` cores at an assumed DRAM -> L1 -> DRAM copy
    rate per core (gbps_per_core, a conservative 10 GB/s: Stage 12's one-core ImagePuller moved 14.3 GB/s PCIe ->
    DRAM); the device unit replaces the assumption with a measurement. Every byte is read once and written once."""
    h_r = 64 // tp
    if kind == "sparse_dsa":
        lat = t_rows * KV_LORA * 2  # the replicated latent cache: every chip writes every row
        pools = (t_rows // KPOOL) * INDEX_DIM * 2
        ring = n_seg * RING_TILES * TILE_BF16
        moved = 2 * (lat + pools + ring)
        return dict(latent_bytes=lat, pool_bytes=pools, ring_bytes=ring, written=lat + pools + ring, moved=moved,
                    span_us=moved / (writers * gbps_per_core * 1e3))  # fmt: skip
    n_s, w_r = 16 * h_r, 24576 // tp
    n_c = 3 * (w_r // TILE)
    s_bytes = n_seg * n_s * TILE_FP32
    c_tiles = n_seg * n_c * TILE_BF16  # read + modify + write of every conv tile per segment
    moved = 2 * s_bytes + 3 * c_tiles  # C: scratch read, stack read, stack write
    return dict(state_bytes=s_bytes, conv_rows_bytes=n_seg * n_c * 64, conv_rmw_bytes=c_tiles, written=s_bytes + c_tiles,
                moved=moved, span_us=moved / (writers * gbps_per_core * 1e3))  # fmt: skip


# ---------------------------------------------------------------------------------------------------- descriptors
def _rt(rtd: dict):
    rt = ttnn.RuntimeArgs()
    for (x, y), args in rtd.items():
        rt[x][y] = [int(a) for a in args]
    return rt


def _crs(cores: list):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _acc(t) -> list:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _split(n: int, cores: list) -> list:
    per = -(-n // max(1, len(cores)))
    return [(min(i * per, n), max(0, min(per, n - min(i * per, n)))) for i in range(len(cores))]


def _cb(i: int, size: int, crs):
    return ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=size)])  # fmt: skip


def _dm(proc):
    p = ttnn.DataMovementProcessor.RISCV_1 if proc == 1 else ttnn.DataMovementProcessor.RISCV_0
    noc = ttnn.NOC.RISCV_1_default if proc == 1 else ttnn.NOC.RISCV_0_default
    return ttnn.DataMovementConfigDescriptor(processor=p, noc=noc)


def _kernel(src: str, cores: list, ct: list, rtd: dict, defines: list, proc: int):
    return ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, src),
                                 source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=_crs(cores),
                                 compile_time_args=[int(v) for v in ct], runtime_args=_rt(rtd), defines=defines,
                                 config=_dm(proc))  # fmt: skip


def _blocks(device, src_blocks, src_rows: int) -> tuple:
    """src_blocks = (L1 address, [logical (x, y) of the MM core holding row block j]) -> (mode, addr, NoC pairs)"""
    if src_blocks is None:
        return 0, None, []
    addr, cores = src_blocks
    assert len(cores) >= src_rows // TILE, (len(cores), src_rows)
    xy = []
    for x, y in cores[: src_rows // TILE]:
        n = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        xy += [int(n.x), int(n.y)]
    return 1, int(addr), xy


def latent_parts(src, cache, page_table, segtab, cores: list, *, t_rows: int, src_row0: int = 0,
                 src_rows: Optional[int] = None, wait=(SEM_LAT, WAIT_NONE, 0), cb_scr: int = CB_SCR, depth: int = 16,
                 nseg_max: int = NSEG_MAX, write_null: bool = False, proc: int = 1, defines=(), src_blocks=None):  # fmt: skip
    """(kernels, cbs) of the latent page writes on `cores` (logical (x, y)), the T-row pages split across them.
    src [1, 1, T(or local rows), 512] bf16 TILE (the TP2 DRAM store; with src_blocks = (l1_addr, [MM core of row
    block j]) the rows are read from the §4b L1 row blocks instead and src is any tensor); cache [num_blocks, 1, 64,
    512] bf16 TILE; page_table [rows, bps] int32 ROW_MAJOR; segtab = SegTable.tensor(). wait = (semaphore id,
    WAIT_*, count)."""
    src_rows = t_rows if src_rows is None else src_rows
    assert int(cache.shape[-1]) == KV_LORA and int(cache.shape[-2]) == BLOCK
    assert src_blocks is not None or int(src.shape[-1]) == KV_LORA
    pt_bytes = int(page_table.shape[-1]) * 4
    sem, mode, count = wait
    smode, saddr, xy = _blocks(cache.device(), src_blocks, src_rows)
    ct = [cb_scr, KX, BLOCK, sem, mode, count, nseg_max, pt_bytes, depth, smode] + _acc(src) + _acc(cache) \
        + _acc(page_table) + _acc(segtab)  # fmt: skip
    head = [saddr if smode else src.buffer_address(), cache.buffer_address(), page_table.buffer_address(),
            segtab.buffer_address(), src_row0, src_rows]  # fmt: skip
    rtd = {xy_: head + list(fc) + xy for xy_, fc in zip(cores, _split(t_rows // BLOCK, cores))}
    defs = list(defines) + ([("WRITE_NULL", "1")] if write_null else [])
    size = depth * TILE_BF16 + _a64(table_bytes(nseg_max)) + nseg_max * _a64(pt_bytes)
    return [_kernel("cache_lat.cpp", cores, ct, rtd, defs, proc)], [_cb(cb_scr, size, _crs(cores))]


def dsa_parts(pooled, store, kg, ring, segtab, cores: list, *, t_rows: int, pages_per_slot: int, src_row0: int = 0,
              src_rows: Optional[int] = None, wait=(SEM_KG, WAIT_NONE, 0), cb_scr: int = CB_SCR,
              nseg_max: int = NSEG_MAX, proc: int = 1, defines=(), src_blocks=None):  # fmt: skip
    """(kernels, cbs) of the pooled-key page writes (split across `cores`) + every segment's ring (cores[0]).
    pooled [1, 1, T / 4, 128] bf16 TILE; store [B x pps, 1, 32, 128] bf16 TILE; kg [1, 1, T, 256] bf16 TILE; ring
    [B, 1, 32, 256] bf16 TILE. src_blocks = (kg_l1_addr, pooled_l1_addr, [MM core of row block j]): the §4b L1
    row-block source (kg block j on core j; pooled page j' on the core of row block 4 j')."""
    src_rows = t_rows if src_rows is None else src_rows
    assert int(store.shape[-1]) == INDEX_DIM and int(ring.shape[-1]) == KG_W
    assert src_blocks is not None or (int(pooled.shape[-1]) == INDEX_DIM and int(kg.shape[-1]) == KG_W)
    sem, mode, count = wait
    smode, kaddr, xy = _blocks(store.device(), None if src_blocks is None else (src_blocks[0], src_blocks[2]), src_rows)
    paddr = int(src_blocks[1]) if src_blocks is not None else pooled.buffer_address()
    ct = [cb_scr, sem, mode, count, nseg_max, POOL_TILES, RING_TILES, pages_per_slot, RING_TILES, smode] \
        + _acc(pooled) + _acc(store) + _acc(kg) + _acc(ring) + _acc(segtab)  # fmt: skip
    head = [paddr, store.buffer_address(), kaddr if smode else kg.buffer_address(), ring.buffer_address(),
            segtab.buffer_address(), src_row0, src_rows]  # fmt: skip
    rtd = {xy_: head + list(fc) + [1 if i == 0 else 0] + xy
           for i, (xy_, fc) in enumerate(zip(cores, _split(t_rows // POOL_ROWS, cores)))}  # fmt: skip
    size = max(POOL_TILES, 2 * RING_TILES) * TILE_BF16 + _a64(table_bytes(nseg_max))
    return [_kernel("cache_dsa.cpp", cores, ct, rtd, list(defines), proc)], [_cb(cb_scr, size, _crs(cores))]


def state_parts(scratches: list, stacks: list, segtab, cores: list, *, chunk: int, wait=(SEM_STATE, WAIT_NONE, 0),
                cb_scr: int = CB_SCR, nseg_max: int = NSEG_MAX, scr_dram: bool = True, proc: int = 0, defines=()):  # fmt: skip
    """(kernels, cbs) of P9: scratches [(S [1, H_r, 128, 128] fp32, C [1, 3, 1, W_r] bf16)] per segment (<= nseg_max;
    missing segments reuse the last), stacks [(S_c [chunk, H_r, 128, 128], C_c [1, 3, chunk, W_r])] per state chunk.
    S pages and conv tiles are split across `cores`."""
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import _pages

    assert 0 < len(scratches) <= nseg_max and stacks, (len(scratches), len(stacks))
    n_s, n_c = _pages(scratches[0][0]), _pages(scratches[0][1])
    for S, C in scratches:
        assert (_pages(S), _pages(C)) == (n_s, n_c)
    for S, C in stacks:
        assert _pages(S) == n_s * chunk and _pages(C) == n_c, (n_s, n_c, chunk)
    sem, mode, count = wait
    ct = [cb_scr, sem, mode, count, nseg_max, n_s, n_c, chunk, len(stacks), int(scr_dram)] + _acc(segtab)
    head = [segtab.buffer_address()]
    for k in range(nseg_max):
        S, C = scratches[min(k, len(scratches) - 1)]
        head += [S.buffer_address(), C.buffer_address()]
    for S, C in stacks:
        head += [S.buffer_address(), C.buffer_address()]
    rtd = {xy: head + list(fs) + list(fc) for xy, fs, fc in zip(cores, _split(n_s, cores), _split(n_c, cores))}
    size = 2 * TILE_FP32 + 2 * TILE_BF16 + _a64(table_bytes(nseg_max))
    return [_kernel("cache_state.cpp", cores, ct, rtd, list(defines), proc)], [_cb(cb_scr, size, _crs(cores))]


def semaphores(cores: list, ids=(SEM_LAT, SEM_KG, SEM_STATE)) -> list:
    """the writers' counters (initial 0; the family resets them once per layer start, §4b)"""
    return [ttnn.SemaphoreDescriptor(id=i, core_ranges=_crs(cores), initial_value=0) for i in ids]


def run(parts: list, io: list):
    """standalone: the parts' kernels + CBs as ONE program (generic_op; captured in a trace by the caller)"""
    kernels, cbs = [], []
    for k, c in parts:
        kernels += k
        cbs += c
    return ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


def free_cores(device, n: int, taken=()) -> list:
    """n worker cores from the top-right of the grid, outside `taken` (standalone runs; the family places the real
    writers on its reader / tail cores)"""
    grid = device.compute_with_storage_grid_size()
    out = []
    for y in reversed(range(int(grid.y))):
        for x in reversed(range(int(grid.x))):
            if (x, y) not in set(taken) and len(out) < n:
                out.append((x, y))
    return out
