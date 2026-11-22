# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_GLUED_ONECALL=on (PB.7B, owner 2026-11-20; bringup/PACKED_DESIGN.md "PB.7B"): a glued chunk's attention ONCE a
layer over every segment, each row at its own segment's positions, slot caches and history (no row moves).

PB.7's path (b) (removed in PB.9G) moved each segment's rows to rows [0, n) and ran today's attention steps once per
segment (two whole-chunk gathers a segment a layer). Here a glued chunk
(tt/packed.GluedChunk: segments back to back at any tile row, a chip may hold rows of several segments and a segment
may span chips) runs the PB.3A one-call path (tt/packed_onecall.OneCall) with every per-chip lane argument replaced by
a per-ROW one:

    rope rows        tt/rope.GluedStarts: each segment's rows at its own positions (RopeTable / CompressRope slices,
                     one memo a chunk), then q_a, kv, q, wq_b, wo_a, wo_b, the compressor's matmuls, the indexer's gates
                     and query once (row-local)
    window region    a glued region a layer [carry 128 | own rows R | one 128-row carry block per segment] (GluedSend:
                     ONE prepared program: each chip's own rows, its last 128 rows to the next chip, every continuing
                     segment's slot carry into its block (read in place from the slot's served snapshot, or the ring
                     carry), every segment's ring pieces into its slot's ring); each row's index list (host, uploaded
                     once a descriptor) reads its own segment's positions only: own rows, the previous chip's rows, or
                     its segment's carry block, ascending (today's lists, mapped)
    compressor       once; each segment's served writes through server-order index rows offset by its model row
                     ((d + off) mod S: the rows path (b)'s move put first), every segment in one row gather (PB.3K)
    indexer          per-chip lanes of the one-call ring (ring_indexer_score_dsa lane_rows = R: each chip its own lane)
                     in PASSES: pass p = each chip's p-th segment; a pass's segments in calls of <= 4 key buffers of
                     one capacity (DS41-328); a chip whose pass segment starts inside it scores its query / gate /
                     block-bias rows rolled so that segment's first row is row 0 (its lane start = that row's
                     position: every causal offset >= 0); each call's top-k (per-chip valid length) and candidate pool
                     copied back to the segment's rows (ChipRows) of one buffer of every row
    sparse_sdpa      kv_lanes (each chip a lane buffer) in the same passes: pass p's rows read their segment's history
                     buffer, the other rows only their window keys (their top-k part sentinel); pass outputs copied
                     onto pass 0's rows of their segment (ChipRows)

Every row computes exactly the bytes path (b) computes for it (same keys in the same order, same scores, same top-k
inputs and valid lengths, same served rows): the oracle (tests/packed/test_glued_oracle.py) with the switch on is
bit-identical. PB.9G (OWNER 2026-11-21, PACKED_CHIPS=on, the shipped value): every packed chunk runs here on
CHIP-ALIGNED LANES (tt/packed.GluedChunk(align = rows a chip): each segment owns whole chips from a chip's row 0), so
every per-row argument above is a per-chip one: one pass, the ring's lane ranges one a chip, no phase rolls or keep
copies (identity piece tables), sparse_sdpa by kv_lanes with the lane map a runtime value (sdpa fork
kv_lane_map_runtime); encoder replay segments run here too (their served writes from write_start, ``_srv`` and the
derive's srv rows; their window carry class 0). PB.7's path (b) and the fixed-lane path are removed: a chunk this path
does not take (PREFILL_END=end, a box with fewer than 128 rows a chip, more than GLUED_SEGS segments, a replay without
DECODER_SKIP=decode) raises (``glued_call_ok``). PACKED_CHIPS=off: PB.9X's back-to-back layout (a chip may hold rows of
two segments: the passes, ChipRows and kv_lane_rows below).

Per-chip tables (row pieces, index lists, keep masks, served-write index rows) are constants of the chunk's descriptor
(the chunk header), made on the host and uploaded once a descriptor (cached), as PB.3A's per-tuple constants.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.common.bringup.testing import profiler

from . import ccl
from .attention import SENTINEL, WINDOW, _upload
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS
from .cores import sub_cores, to_rm
from .packed_onecall import (
    ONE_RING,
    OneCall,
    _has_map_runtime,
    _kbuf,
    _out,
    _own_extent,
    onecall_on,
    sdpa_lanes,
    snapshot_carry_ok,
    topk_rows_lanes,
)
from .page_copy import dyn
from .rope import GluedStarts
from .settings import settings

GLUED_SEGS = 6  # segments a glued chunk (tt/packed.GLUED_MAX); one carry block and one ring slot each
RING_PIECES = 4  # ring pieces a segment (tt/window_send.RING_PIECES)
MAX_PIECES = 8  # ChipRows: row pieces a table
TABLE_WORDS = 32  # ChipRows: words a table (count + 3 x MAX_PIECES)
N_TABLES = 64  # ChipRows: tables a chunk


KV_ROWS_MAX = 8  # sparse_sdpa kv_lane_rows: row ranges a chip (the fork's KV_ROWS_MAX)


def _has_lane_rows() -> bool:
    """The sdpa fork's kv_lane_rows option is built (ttnn.bringup.sparse_sdpa's binding names it)."""
    return "kv_lane_rows" in (getattr(ttnn.bringup.sparse_sdpa, "__doc__", "") or "")


LANE_ROWS = [_has_lane_rows()]  # (a list: tests may switch it off for the passes)


def _max_bufs() -> int:
    """ring_indexer_score_dsa lanes: key buffers a call (8 since the fork's PB.7B change, its binding says so; else 4)."""
    doc = getattr(ttnn.bringup.ring_indexer_score_dsa, "__doc__", "") or ""
    return 8 if "(up to 8)" in doc else 4


MAX_BUFS = [_max_bufs()]  # (a list: tests may lower it)
COPY_CORES = (11, 8)  # ChipRows: a [x, y] rectangle of cores a chip from (0, 0) (data movement only; inside the 11 x 10
# compute sub-device of the gather hide)
BATCH_BYTES = 32768  # ChipRows: L1 bytes of the page pieces read a NoC barrier
SEG_BYTES = 8192  # ChipRows: bytes a piece of a page at most (a logits row is up to T x 2 bytes)


def dynamic_on() -> bool:
    """PACKED_GLUED_DYNAMIC=on (PB.7C, call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED_GLUED_DYNAMIC") == "on"


def _has_ranges() -> bool:
    """The indexer fork's lane_ranges option is built (ttnn.bringup.ring_indexer_score_dsa's binding names it)."""
    return "lane_ranges" in (getattr(ttnn.bringup.ring_indexer_score_dsa, "__doc__", "") or "")


RANGES = [_has_ranges()]  # (a list: tests may switch it off)
RANGES_MAX = 8  # ring_indexer_score_dsa lane_ranges: ranges a chip (the fork's kRangesMax)
RANGE_STRIDE = 1 + 5 * RANGES_MAX  # words a chip


def glued_onecall_on() -> bool:
    """PACKED_GLUED_ONECALL=on (call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED_GLUED_ONECALL") == "on"


def glued_call_ok(model, desc, segs: list) -> str | None:
    """None when a glued chunk runs here (GluedCall), else why it keeps path (b)."""
    if not (glued_onecall_on() and onecall_on()):
        return "switch off"
    if not segs:
        return "no segment computes"
    if len(desc.segments) > GLUED_SEGS:
        return f"{len(desc.segments)} segments"
    if settings.get("PREFILL_END") != "cap":
        return "PREFILL_END=end"
    sp, tp = int(model.mesh.shape[SP_AXIS]), int(model.mesh.shape[TP_AXIS])
    if model.chunk // (sp * tp) < WINDOW:
        return "fewer than 128 rows a chip"
    if not ONE_RING[0]:
        return "the indexer fork's lanes ring is not built"
    if getattr(desc, "align", 0) and int(desc.align) != model.chunk // (sp * tp):
        return f"chip-aligned lanes of {desc.align} rows on a mesh of {model.chunk // (sp * tp)} rows a chip"
    for r in segs:
        if r.seg.replay and getattr(model, "skip_mode", None) != "decode":
            return "an encoder replay segment needs DECODER_SKIP=decode"
        if (r.run_end - r.start) % 32:
            return "a segment's rows are not whole tiles"
    return None


# ---------------------------------------------------------------------------------------------- geometry
class GluedGeo:
    """The rows of a glued chunk on a mesh: chip q (chip-linear r TP + c) holds model rows [q R, (q + 1) R); segment j
    (``segs[j]``, a computing tt/packed_device.SegRun, desc order) holds model rows [off[j], off[j] + rows[j]).
    ``chip[q]`` = [(j, a, b)]: segment j's rows on chip q are its local rows [a, b), in row order; pass p of chip q =
    its p-th entry."""

    def __init__(self, mesh, desc, segs: list):
        self.mesh = mesh
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        self.n = self.sp * self.tp
        self.chunk = int(desc.chunk)
        self.R = self.chunk // self.n
        self.segs = list(segs)
        self.off = [int(desc.lane_off(r.lane)) for r in segs]
        self.rows = [int(r.run_end - r.start) for r in segs]
        self.used = int(desc.used_rows())
        assert all(o % 32 == 0 and k % 32 == 0 for o, k in zip(self.off, self.rows)), (self.off, self.rows)
        self.chip = [[] for _ in range(self.n)]
        for j, (o, k) in enumerate(zip(self.off, self.rows)):
            for q in range(o // self.R, (o + k - 1) // self.R + 1):
                a, b = max(o, q * self.R) - q * self.R, min(o + k, (q + 1) * self.R) - q * self.R
                self.chip[q].append((j, a, b))
        self.passes = max(1, max(len(c) for c in self.chip))
        self.align = int(getattr(desc, "align", 0))
        if self.align:  # PB.9G (chip-aligned lanes): every chip holds at most one segment, from its first row
            assert self.passes == 1, self.chip
            assert all(o % self.R == 0 for o in self.off), (self.off, self.R)
        pieces = []  # (the rope rows of every model row in order: a segment's computed rows at its positions; pad rows
        # (PB.9G: a segment's last chip past its rows, an idle segment's chip; the tail) at positions 0.., never read)
        by_lane = {int(r.lane): (int(r.start), k) for r, k in zip(segs, self.rows)}
        for k in range(desc.lanes):
            st, n = by_lane.get(k, (0, 0))
            if n:
                pieces.append((st, n))
            pad = (desc.ext(k) - n) if self.align else 0
            if pad:
                pieces.append((0, pad))
        if self.used < self.chunk:
            pieces.append((0, self.chunk - self.used))
        self.starts = GluedStarts(pieces)

    def key(self) -> tuple:
        return (self.chunk, self.n, tuple(self.off), tuple(self.rows), tuple(int(r.start) for r in self.segs))

    def entry(self, q: int, p: int):
        c = self.chip[q]
        return c[p] if p < len(c) else None

    def position(self, j: int, q: int, a: int) -> int:
        """The position of chip q's local row a, a row of segment j."""
        return int(self.segs[j].start) + q * self.R + a - self.off[j]

    # ------------------------------------------------------------------ host tables
    def keep_rows(self) -> torch.Tensor:
        """[SP, TP, R, 1]: 1.0 on every chip's rows below the used rows (the tail pad 0)."""
        t = torch.zeros(self.chunk)
        for o, k in zip(self.off, self.rows):  # (PB.9G: a segment's computed rows; its last chip's pad 0)
            t[o : o + k] = 1.0
        return t.view(self.sp, self.tp, self.R, 1)

    def region_lists(self, classes: tuple, seg_block: list) -> torch.Tensor:
        """[SP, TP, R, 128] int32: each row's window index list in the glued region [carry 128 | own rows R |
        GLUED_SEGS carry blocks of 128]: the positions max(p - 127, start - cls) .. p of its own segment, ascending
        (tt/attention.WindowTables.region_lists' lists, per row): a row of the chip's own rows at 128 + row, one of the
        previous chip's last 128 rows (the neighbour carry) at row - (R - 128), a position before the segment's start
        in the segment's carry block (``seg_block[j]``) at 128 + R + 128 block + 128 + (position - start). Pad rows:
        their own row only."""
        R, n = self.R, self.n
        jv = torch.arange(WINDOW)
        out = torch.full((n, R, WINDOW), SENTINEL, dtype=torch.int64)
        own = torch.arange(R)
        out[:, :, 0] = WINDOW + own.view(1, R)  # (pad rows; overwritten for every segment row)
        for j, (o, k) in enumerate(zip(self.off, self.rows)):
            cls = int(classes[j])
            d = torch.arange(k)
            m = o + d  # model rows
            q, i = m // R, m % R
            x0 = torch.clamp(d - (WINDOW - 1), min=-cls)  # first position offset (relative to start)
            X = x0.view(-1, 1) + jv.view(1, -1)  # [k, 128]
            valid = X <= d.view(-1, 1)
            mx = o + X
            q0 = (q * R).view(-1, 1)
            r_own = WINDOW + mx - q0
            r_prev = mx - (q0 - WINDOW)
            r_carry = WINDOW + R + WINDOW * int(seg_block[j]) + WINDOW + X
            ids = torch.where(X < 0, r_carry, torch.where(mx >= q0, r_own, r_prev))
            assert bool((ids[valid] >= 0).all()), "a window row two chips back"
            ids = torch.where(valid, ids, torch.full_like(ids, SENTINEL))
            out[q, i] = ids
        return out.view(self.sp, self.tp, R, WINDOW).to(torch.int32)


# ---------------------------------------------------------------------------------------------- ChipRows
_ROWS_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);     // bytes a page
constexpr uint32_t PPU = get_compile_time_arg_val(1);      // pages a unit (a row: 1; a tile row: its tiles)
constexpr uint32_t B = get_compile_time_arg_val(2);        // pieces a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(3);       // cores
constexpr uint32_t NBLK = get_compile_time_arg_val(4);     // blocks (the dims before the rows)
constexpr uint32_t SSTRIDE = get_compile_time_arg_val(5);  // source pages a block
constexpr uint32_t DSTRIDE = get_compile_time_arg_val(6);  // destination pages a block
constexpr uint32_t SEGB = get_compile_time_arg_val(7);     // bytes a piece of a page (a multiple of 64, <= PAGE)
constexpr auto s_args = TensorAccessorArgs<8>();
constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
constexpr auto d_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t cst = get_arg_val<uint32_t>(1);
    const uint32_t dst = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);       // this core's index of NC
    const uint32_t toff = get_arg_val<uint32_t>(4);    // this call's table: byte offset in the chip's table page
    const uint32_t cbytes = get_arg_val<uint32_t>(5);  // bytes copied of each page (its prefix; PAGE: whole)
    const uint32_t cpage = get_arg_val<uint32_t>(6);   // bytes of the table page
    const auto S = TensorAccessor(s_args, src, PAGE);
    const auto C = TensorAccessor(c_args, cst, cpage);
    const auto D = TensorAccessor(d_args, dst, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t tab = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t buf = tab + 128;
    noc_async_read(C.get_noc_addr(0, toff), tab, 128);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    const uint32_t np = w[0] < 8 ? w[0] : 8;
    uint32_t units = 0;
    for (uint32_t p = 0; p < np; ++p) {
        units += w[3 + 3 * p];
    }
    const uint32_t nb_bytes = cbytes < PAGE ? cbytes : PAGE;
    const uint32_t nseg = (nb_bytes + SEGB - 1) / SEGB;
    const uint32_t per_blk = units * PPU * nseg;
    const uint32_t total = NBLK * per_blk;
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(total) * k) / NC);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(total) * (k + 1)) / NC);
    uint64_t daddr[B];
    uint32_t dlen[B];
    uint32_t g = lo;
    while (g < hi) {
        uint32_t nb = 0;
        for (; nb < B && g < hi; ++nb, ++g) {
            const uint32_t blk = g / per_blk, rem = g % per_blk;
            const uint32_t pg = rem / nseg, sg = rem % nseg;
            const uint32_t u = pg / PPU, pp = pg % PPU;
            uint32_t base = 0, su = 0, du = 0;
            for (uint32_t p = 0; p < np; ++p) {
                const uint32_t s0 = w[1 + 3 * p], d0 = w[2 + 3 * p], cnt = w[3 + 3 * p];
                if (u < base + cnt) {
                    su = s0 + (u - base);
                    du = d0 + (u - base);
                    break;
                }
                base += cnt;
            }
            const uint32_t off = sg * SEGB;
            const uint32_t len = (nb_bytes - off) < SEGB ? (nb_bytes - off) : SEGB;
            dlen[nb] = len;
            daddr[nb] = D.get_noc_addr(blk * DSTRIDE + du * PPU + pp, off);
            noc_async_read(S.get_noc_addr(blk * SSTRIDE + su * PPU + pp, off), buf + nb * SEGB, len);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < nb; ++t) {
            noc_async_write(buf + t * SEGB, daddr[t], dlen[t]);
        }
        noc_async_write_barrier();
    }
}
"""


def _layout_geo(t: ttnn.Tensor) -> tuple:
    """(page bytes, pages a unit, rows a unit, blocks, pages a block) of a 4-D DRAM tensor [1, B, S, W]: a unit is a row
    (ROW_MAJOR) or a tile row (TILE)."""
    s = [int(v) for v in t.shape]
    assert len(s) == 4, s
    blocks, rows, width = s[0] * s[1], s[2], s[3]
    if t.layout == ttnn.TILE_LAYOUT:
        assert rows % 32 == 0 and width % 32 == 0, s
        ppu = width // 32
        return int(ttnn.tile_size(t.dtype)), ppu, 32, blocks, (rows // 32) * ppu
    eb = {ttnn.bfloat16: 2, ttnn.uint32: 4, ttnn.float32: 4, ttnn.int32: 4, ttnn.uint8: 1, ttnn.uint16: 2}[t.dtype]
    return width * eb, 1, 1, blocks, rows


class ChipTables:
    """The per-chip row-piece tables of one glued chunk: table t = [count, (source unit, destination unit, units) x
    count] of each chip, made on the host, uploaded as ONE [SP, TP] sharded uint32 page a chip (re-uploaded only when a
    table was added since)."""

    def __init__(self, mesh, n: int):
        self.mesh, self.n = mesh, n
        self.words = torch.zeros(n, N_TABLES * TABLE_WORDS, dtype=torch.int32)
        self.index: dict = {}
        self.dev = None
        self.old = []  # (tables replaced in this chunk: freed at its end, never under a queued reader)
        self.dirty = True

    def add(self, key, pieces: list, unit: int) -> int:
        """``pieces`` per chip [(source row, destination row, rows)] -> table index (units of ``unit`` rows)."""
        k = (key, unit)
        hit = self.index.get(k)
        if hit is not None:
            return hit
        t = len(self.index)
        assert t < N_TABLES, "ChipTables: tables a chunk"
        for q, ps in enumerate(pieces):
            ps = [p for p in ps if p[2] > 0]
            assert len(ps) <= MAX_PIECES, ps
            w = [len(ps)]
            for s, d, c in ps:
                assert s % unit == 0 and d % unit == 0 and c % unit == 0, (s, d, c, unit)
                w += [s // unit, d // unit, c // unit]
            self.words[q, t * TABLE_WORDS : t * TABLE_WORDS + len(w)] = torch.tensor(w, dtype=torch.int32)
        self.index[k] = t
        self.dirty = True
        return t

    def tensor(self) -> ttnn.Tensor:
        if self.dirty or self.dev is None:
            if self.dev is not None:
                self.old.append(self.dev)
            sp, tp = int(self.mesh.shape[SP_AXIS]), int(self.mesh.shape[TP_AXIS])
            w = self.words.view(sp, tp, 1, N_TABLES * TABLE_WORDS)
            self.dev = _upload(self.mesh, w, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
            self.dirty = False
        return self.dev

    def free(self) -> None:
        for t in self.old + [self.dev]:
            if t is not None and t.is_allocated():
                dealloc(t)
        self.dev, self.old = None, []


class ChipRows:
    """One prepared program: pages of ``src`` -> ``dst`` by a per-chip table of row pieces (ChipTables), every block
    (the dims before the rows) alike. Data movement only, COPY_CORES a chip."""

    def __init__(self, mesh, src: ttnn.Tensor, dst: ttnn.Tensor, cst: ttnn.Tensor):
        self.mesh = mesh
        self.page, self.ppu, self.unit, self.blocks, self.sstride = _layout_geo(src)
        page, ppu, unit, blocks, self.dstride = _layout_geo(dst)
        assert (page, ppu, unit, blocks) == (self.page, self.ppu, self.unit, self.blocks), (src.shape, dst.shape)
        self.segb = min(-(-self.page // 64) * 64, SEG_BYTES)
        self.batch = max(1, min(32, BATCH_BYTES // self.segb))
        x, y = COPY_CORES
        self.cores = [(i, j) for j in range(y) for i in range(x)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))])
        self.cpage = int(cst.shape[-1]) * 4
        self._pp = None

    def _build(self, src, cst, dst):
        ct = [self.page, self.ppu, self.batch, len(self.cores), self.blocks, self.sstride, self.dstride, self.segb]
        for t in (src, cst, dst):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 64 + 128 + self.batch * self.segb
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=size, core_ranges=self.crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
            )
        ]
        rt = ttnn.RuntimeArgs()
        a = [int(src.buffer_address()), int(cst.buffer_address()), int(dst.buffer_address())]
        for k, (cx, cy) in enumerate(self.cores):
            rt[cx][cy] = a + [k, 0, 0, self.cpage]
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_ROWS_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        sp, tp = tuple(self.mesh.shape)
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(sp - 1, tp - 1))] = prog
        return md

    def __call__(self, src: ttnn.Tensor, dst: ttnn.Tensor, cst: ttnn.Tensor, table: int, cbytes=None) -> None:
        io = [src, cst, dst]
        if self._pp is None:
            self._pp = ttnn.bringup.prepare_program(self._build(src, cst, dst), io, [(0, 4), (0, 5)])
            want = len(self.cores) * 5  # (src, cst, dst, the table offset and the byte count a core)
            assert self._pp.patch_count == want, f"ChipRows: {self._pp.patch_count} patched slots, {want} expected"
        nb = self.page if cbytes is None else max(64, min(self.page, -(-int(cbytes) // 64) * 64))
        ttnn.bringup.run_prepared(self._pp, io, [int(table) * TABLE_WORDS * 4, nb])


_ROWS: dict = {}


def chip_rows(src: ttnn.Tensor, dst: ttnn.Tensor, tables: ChipTables, key, pieces: list, cbytes=None) -> None:
    """dst's rows <- src's rows by ``pieces`` (per chip [(source row, destination row, rows)], every block alike);
    ``key`` names the pieces in ``tables`` (made once a chunk); ``cbytes``: only that prefix of each page (a row's
    first columns of a ROW_MAJOR tensor; None: whole pages)."""
    t = tables.add(key, pieces, _layout_geo(src)[2])
    chip_rows_at(src, dst, tables.tensor(), t, cbytes)


def chip_rows_at(src: ttnn.Tensor, dst: ttnn.Tensor, cst: ttnn.Tensor, t: int, cbytes=None) -> None:
    """chip_rows with table ``t`` of a table page already on the device (``cst``: ChipTables' layout; PB.7C attempt
    3: tt/glued_derive's derived piece tables, in rows)."""
    mesh = src.device()
    geo = _layout_geo(src)
    ck = (
        id(mesh),
        geo,
        _layout_geo(dst),
        tuple(ttnn.TensorAccessorArgs(src).get_compile_time_args()),
        tuple(ttnn.TensorAccessorArgs(dst).get_compile_time_args()),
        tuple(cst.shape),
    )
    hit = _ROWS.get(ck)
    if hit is None or hit.mesh is not mesh:
        hit = _ROWS[ck] = ChipRows(mesh, src, dst, cst)
    hit(src, dst, cst, t, cbytes)


def _rows_in_rows(src: ttnn.Tensor, dst: ttnn.Tensor, cst: ttnn.Tensor, t: int, cbytes=None) -> None:
    """chip_rows_at with a derived table (tt/glued_derive: pieces in rows, so a ROW_MAJOR source: one row a unit)."""
    assert _layout_geo(src)[2] == 1, "derived piece tables count rows: a ROW_MAJOR source"
    chip_rows_at(src, dst, cst, t, cbytes)


# ---------------------------------------------------------------------------------------------- GluedSend
_GSEND_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#if G_SEND
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#endif

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page (a row)
constexpr uint32_t SEG = get_compile_time_arg_val(1);   // bytes a packet (whole 64 B lines)
constexpr uint32_t NSEG = get_compile_time_arg_val(2);  // packets a page
constexpr uint32_t NP = get_compile_time_arg_val(3);    // rows a carry (128)
constexpr uint32_t B = get_compile_time_arg_val(4);     // pages read a NoC barrier
constexpr uint32_t Q0 = get_compile_time_arg_val(5);    // the first of the chip's own rows sent (its last NP)
constexpr uint32_t C0 = get_compile_time_arg_val(6);    // the region row of carry block 0 (128 + R)
constexpr uint32_t NS = get_compile_time_arg_val(7);    // segments (carry blocks, tail tensors)
constexpr uint32_t SLOT = (SEG + 63) & ~63u;
constexpr auto r_args = TensorAccessorArgs<8>();
constexpr auto o_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
constexpr auto t0_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
constexpr auto t2_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
constexpr auto t3_args = TensorAccessorArgs<t2_args.next_compile_time_args_offset()>();
constexpr auto t4_args = TensorAccessorArgs<t3_args.next_compile_time_args_offset()>();
constexpr auto t5_args = TensorAccessorArgs<t4_args.next_compile_time_args_offset()>();

template <typename Acc, typename Reg>
inline void carry_copy(const Acc& T, const Reg& Rg, uint32_t buf, uint32_t base, uint32_t rot, uint32_t row0) {
    for (uint32_t p0 = 0; p0 < NP; p0 += B) {
        const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
        for (uint32_t s = 0; s < NSEG; ++s) {
            const uint32_t off = s * SEG;
            const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(T.get_noc_addr(base + (p0 + i + rot) % NP, off), buf + i * SLOT, bytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_write(buf + i * SLOT, Rg.get_noc_addr(row0 + p0 + i, off), bytes);
            }
            noc_async_write_barrier();
        }
    }
}

void kernel_main() {
    size_t a = 0;
    const uint32_t region = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t sem = get_arg_val<uint32_t>(a++);  // the arrival counter (same address on every chip)
    const uint32_t px = get_arg_val<uint32_t>(a++);
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    uint32_t tail[6];
    for (uint32_t t = 0; t < 6; ++t) {
        tail[t] = get_arg_val<uint32_t>(a++);
    }
    const size_t v = a;  // per segment: active, tail tensor, first page, rotation (per-run values)
    a += 4 * NS;
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto O = TensorAccessor(o_args, rows, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
#if G_SEND
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the glued window send needs a 2D fabric (agent rule 4)"
#endif
    conn.open();
    const uint64_t arrival = get_noc_addr(px, py, sem);
    constexpr uint32_t total = NP * NSEG;
    uint32_t k = 0;
    for (uint32_t p0 = 0; p0 < NP; p0 += B) {
        const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
        for (uint32_t s = 0; s < NSEG; ++s) {
            const uint32_t off = s * SEG;
            const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(O.get_noc_addr(Q0 + p0 + i, off), buf + i * SLOT, bytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint64_t dst = R.get_noc_addr(p0 + i, off);  // the receiver's region (same buffer address)
                if (++k == total) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, arrival, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(buf + i * SLOT, bytes, reinterpret_cast<uint32_t>(hdr));
                noc_async_writes_flushed();
            }
        }
    }
    conn.close();
#else
    (void)px;
    (void)py;
    (void)mesh_id;
    (void)chip_id;
#endif
    // every continuing segment's slot carry into its block (the same rows on every chip)
    for (uint32_t s = 0; s < NS; ++s) {
        if (get_arg_val<uint32_t>(v + 4 * s) == 0) {
            continue;
        }
        const uint32_t t = get_arg_val<uint32_t>(v + 4 * s + 1);
        const uint32_t base = get_arg_val<uint32_t>(v + 4 * s + 2);
        const uint32_t rot = get_arg_val<uint32_t>(v + 4 * s + 3);
        const uint32_t row0 = C0 + s * NP;
        if (t == 0) {
            carry_copy(TensorAccessor(t0_args, tail[0], PAGE), R, buf, base, rot, row0);
        } else if (t == 1) {
            carry_copy(TensorAccessor(t1_args, tail[1], PAGE), R, buf, base, rot, row0);
        } else if (t == 2) {
            carry_copy(TensorAccessor(t2_args, tail[2], PAGE), R, buf, base, rot, row0);
        } else if (t == 3) {
            carry_copy(TensorAccessor(t3_args, tail[3], PAGE), R, buf, base, rot, row0);
        } else if (t == 4) {
            carry_copy(TensorAccessor(t4_args, tail[4], PAGE), R, buf, base, rot, row0);
        } else {
            carry_copy(TensorAccessor(t5_args, tail[5], PAGE), R, buf, base, rot, row0);
        }
    }
#if G_RECV
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait_min(arrived, 1);
    noc_semaphore_inc(get_noc_addr(sem), 0u - 1u);
    noc_async_atomic_barrier();
#endif
}
"""

_GRING_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);   // bytes a row (a page)
constexpr uint32_t B = get_compile_time_arg_val(1);      // rows read a NoC barrier
constexpr uint32_t RP = get_compile_time_arg_val(2);     // ring pieces a segment
constexpr uint32_t NS = get_compile_time_arg_val(3);     // segments
constexpr uint32_t QR = get_compile_time_arg_val(4);     // the chip's own rows
constexpr uint32_t W0 = get_compile_time_arg_val(5);     // the region row of own row 0 (128)
constexpr auto o_args = TensorAccessorArgs<6>();
constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();

template <typename Src, typename Dst>
inline void copy_rows(const Src& S, const Dst& D, uint32_t buf, uint32_t s0, uint32_t d0, uint32_t n) {
    for (uint32_t r = 0; r < n; r += B) {
        const uint32_t nb = (n - r) < B ? (n - r) : B;
        for (uint32_t i = 0; i < nb; ++i) {
            noc_async_read(S.get_noc_addr(s0 + r + i), buf + i * PAGE, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < nb; ++i) {
            noc_async_write(buf + i * PAGE, D.get_noc_addr(d0 + r + i), PAGE);
        }
        noc_async_write_barrier();
    }
}

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t region = get_arg_val<uint32_t>(1);
    const auto O = TensorAccessor(o_args, rows, PAGE);
    const auto R = TensorAccessor(r_args, region, PAGE);
    const uint32_t buf = (get_write_ptr(1) + 63) & ~63u;
    copy_rows(O, R, buf, 0, W0, QR);  // the chip's own rows into the region
    for (uint32_t s = 0; s < NS; ++s) {  // every segment's ring pieces into its slot's ring (from this chip's rows)
        const auto G = TensorAccessor(g_args, get_arg_val<uint32_t>(2 + s), PAGE);
        for (uint32_t p = 0; p < RP; ++p) {
            const uint32_t x = 2 + NS + 3 * (s * RP + p);
            const uint32_t n = get_arg_val<uint32_t>(x + 2);
            if (n) {
                copy_rows(O, G, buf, get_arg_val<uint32_t>(x), get_arg_val<uint32_t>(x + 1), n);
            }
        }
    }
}
"""


class GluedSend:
    """The glued window region of a layer in one prepared program on every chip (core (0, 0)): io [region, rows,
    tail_0 .. tail_5, ring_0 .. ring_5]. RISCV_0: chip q's last 128 own rows to chip q + 1's region rows [0, 128)
    (fabric, chip-linear order, a fused arrival increment) and every active segment's carry (tail tensor t, first page,
    rotation: carry row i = page base + (i + rot) % 128) into its block, then (q >= 1) the arrival wait; RISCV_1: the
    chip's own rows into region rows [128, 128 + R) and every segment's ring pieces [(own row, ring row, rows)] into
    its slot's ring."""

    CORE = (0, 0)
    BATCH_BYTES = 16384
    COPY_BATCH = 8

    def __init__(self, mesh, q_rows: int):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan
        from ttnn.bringup.combine_rf.sparse_combine_l1 import route

        from .attention import RING
        from .window_send import _torus, segments

        self.mesh = mesh
        R, C = tuple(mesh.shape)
        self.n = R * C
        self.page = 1024  # a bf16 row of 512
        self.q_rows = int(q_rows)
        assert self.q_rows >= WINDOW, q_rows
        self.seg, self.nseg = segments(self.page, int(ttnn.get_tt_fabric_max_payload_size_bytes()))
        self.slot = -(-self.seg // 64) * 64
        self.batch = max(1, min(WINDOW, self.BATCH_BYTES // self.slot)) if self.nseg == 1 else 1
        chips, _, (wrap_x, wrap_y) = mesh_plan(mesh, 1, _torus(mesh))
        cc = ttnn.CoreCoord(*self.CORE)
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
        self.sem = ttnn.create_global_semaphore(mesh, self.crs, 0, ttnn.BufferType.L1_SMALL)
        sem_addr = int(ttnn.get_global_semaphore_address(self.sem))
        v = mesh.worker_core_from_logical_core(cc)
        node = lambda ch: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))  # noqa: E731
        self._chips = {}  # q: (send, recv, args without the io addresses, link or None)
        for q in range(self.n):
            send, recv = q < self.n - 1, q > 0
            src = divmod(q, C)
            link, args = None, [sem_addr, v.x, v.y, 0, 0]
            if send:
                dst = divmod(q + 1, C)
                hops = route(src, dst, R, C, wrap_x, wrap_y, 0)
                assert hops, (src, dst)
                port = chips[src][hops[0][1]]
                dn = node(dst)
                args = [sem_addr, v.x, v.y, int(dn.mesh_id), int(dn.chip_id)]
                link = (node(src), node(port["peer"]), port["links"][0])
            self._chips[q] = (send, recv, args, link)
        alloc = lambda rows: ttnn.zeros(  # noqa: E731
            [1, 1, rows, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM
        )
        self.scratch_tail = [alloc(WINDOW) for _ in range(GLUED_SEGS)]  # (load time: never read)
        self._tkind = None  # (the scratch's accessor args: the ring-carry temps' kind, PB.9G)
        self.scratch_ring = [alloc(RING) for _ in range(GLUED_SEGS)]
        for t in self.scratch_tail + self.scratch_ring:
            ccl.pin(mesh, t)
        self._pps = {}

    def _build(self, region, rows, tails, rings):
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        cc = ttnn.CoreCoord(*self.CORE)
        x0, y0 = self.CORE
        md = ttnn.MeshProgramDescriptor()
        R, C = tuple(self.mesh.shape)
        ra, oa = int(region.buffer_address()), int(rows.buffer_address())
        ta = [int(t.buffer_address()) for t in tails]
        ga = [int(t.buffer_address()) for t in rings]
        ct0 = [self.page, self.seg, self.nseg, WINDOW, self.batch, self.q_rows - WINDOW, WINDOW + self.q_rows]
        ct0 += [GLUED_SEGS] + acc(region) + acc(rows)
        for t in tails:
            ct0 += acc(t)
        ct1 = [self.page, self.COPY_BATCH, RING_PIECES, GLUED_SEGS, self.q_rows, WINDOW] + acc(rows) + acc(region)
        ct1 += acc(rings[0])
        for q in range(self.n):
            r, c = divmod(q, C)
            rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
            send, recv, args, link = self._chips[q]
            p = ttnn.ProgramDescriptor()
            p.cbs = [
                ttnn.CBDescriptor(
                    total_size=self.batch * self.slot + 64,
                    core_ranges=self.crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, self.batch * self.slot + 64)],
                ),
                ttnn.CBDescriptor(
                    total_size=self.COPY_BATCH * self.page + 64,
                    core_ranges=self.crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint8, self.COPY_BATCH * self.page + 64)],
                ),
            ]
            a = [ra, oa] + list(args) + ta + [0] * (4 * GLUED_SEGS)
            if send:
                a += list(ttnn.setup_fabric_connection(self.mesh, *link, p, cc))
            rt = ttnn.RuntimeArgs()
            rt[cc.x][cc.y] = a
            crt = ttnn.RuntimeArgs()
            crt[x0][y0] = [oa, ra] + ga + [0] * (3 * RING_PIECES * GLUED_SEGS)
            src = f"#define G_SEND {int(send)}\n#define G_RECV {int(recv)}\n" + _GSEND_KERNEL
            dm = lambda pr, nc: ttnn.DataMovementConfigDescriptor(processor=pr, noc=nc)  # noqa: E731
            p.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=src,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=self.crs,
                    compile_time_args=ct0,
                    runtime_args=rt,
                    config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_GRING_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=self.crs,
                    compile_time_args=ct1,
                    runtime_args=crt,
                    config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
                ),
            ]
            md[rng] = p
        return md

    def __call__(self, region, rows, tails: list, rings: list, pieces: list, carries: list, anchor=None) -> None:
        """region [1, 1, 128 + R + 128 GLUED_SEGS, 512] bf16 RM (the layer's glued region), rows [1, 1, R, 512] bf16 RM
        (each chip its own masked rows); per segment slot s < GLUED_SEGS: ``tails[s]`` the tensor its carry is read
        from (None: none), ``carries[s]`` (first page, rotation) or None (no carry: its block is not written),
        ``rings[s]`` its slot's ring (None: a scratch), ``pieces[s]`` its ring pieces [(own row, ring row, rows)]."""
        S = GLUED_SEGS
        assert int(rows.shape[-2]) == self.q_rows and int(region.shape[-2]) == WINDOW + self.q_rows + S * WINDOW, (
            rows.shape,
            region.shape,
        )
        rings = [t if t is not None else self.scratch_ring[s] for s, t in enumerate((list(rings) + [None] * S)[:S])]
        utails, tsel = [], []
        for s in range(S):  # (one io entry a distinct tensor; the rest of the 6 tail slots scratch tensors)
            t = tails[s] if s < len(tails) else None
            if t is None or carries[s] is None:
                tsel.append(None)
                continue
            j = next((u for u, x in enumerate(utails) if x.buffer_address() == t.buffer_address()), None)
            if j is None:
                j = len(utails)
                utails.append(t)
            tsel.append(j)
        # PB.9G (DS41-360): one program for every mix of carry sources. The tails' accessors are of two kinds (the
        # served snapshot tensor, ND-sharded or interleaved; an interleaved ring-carry temp, the scratch's kind): the
        # snapshot-kind ones first, then the temp-kind ones, then scratch, and at least one snapshot-kind slot
        # (``anchor``: a snapshot tensor of the chunk, read by no segment when none carries from it)
        akey = lambda t: tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        if getattr(self, "_tkind", None) is None:  # (DS41-353: an instance made before a reload has none)
            self._tkind = akey(self.scratch_tail[0])
        tkind = self._tkind
        keys = [akey(x) for x in utails]
        snaps = [u for u, kk in enumerate(keys) if kk != tkind]
        if not snaps and anchor is not None and len(utails) < S:
            ak = akey(anchor)
            if ak != tkind:
                utails.insert(0, anchor)
                keys.insert(0, ak)
                tsel = [None if j is None else j + 1 for j in tsel]
                snaps = [0]
        order = snaps + [u for u in range(len(utails)) if u not in snaps]
        remap = {u: i for i, u in enumerate(order)}
        utails = [utails[u] for u in order]
        keys = [keys[u] for u in order]
        tsel = [None if j is None else remap[j] for j in tsel]
        k = 0
        while len(utails) < S:
            utails.append(self.scratch_tail[k])
            keys.append(tkind)
            k += 1
        assert len(utails) == S, len(utails)
        io = [region, rows] + utails + rings
        key = tuple(keys)
        pp = self._pps.get(key)
        if pp is None:
            slots = [(0, 7 + S + i) for i in range(4 * S)] + [(1, 2 + S + i) for i in range(3 * RING_PIECES * S)]
            pp = self._pps[key] = ttnn.bringup.prepare_program(self._build(region, rows, utails, rings), io, slots)
            want = self.n * (2 + S + 4 * S + 2 + S + 3 * RING_PIECES * S)
            assert pp.patch_count == want, f"glued send: {pp.patch_count} patched slots, {want} expected"
        vals = []
        for s in range(S):
            c = carries[s] if s < len(carries) else None
            if c is None or tsel[s] is None:
                vals += [0, 0, 0, 0]
            else:
                vals += [1, int(tsel[s]), int(c[0]), int(c[1])]
        for s in range(S):
            ps = list(pieces[s]) if s < len(pieces) and pieces[s] else []
            assert len(ps) <= RING_PIECES, ps
            v = []
            for a, b, n in ps:
                v += [int(a), int(b), int(n)]
            vals += v + [0] * (3 * RING_PIECES - len(v))
        ttnn.bringup.run_prepared(pp, io, vals)


_GSENDS: dict = {}


def glued_send(mesh, q_rows: int) -> GluedSend:
    key = (id(mesh), int(q_rows))
    hit = _GSENDS.get(key)
    if hit is None or hit.mesh is not mesh:
        hit = _GSENDS[key] = GluedSend(mesh, q_rows)
    return hit


# ---------------------------------------------------------------------------------------------- per-descriptor constants
_GCONST: dict = {}  # (id(mesh), kind, key) -> (mesh, tensor): constants of a descriptor, kept (bounded)
_GCONST_MAX = 64


def _gconst(mesh, kind: str, key, make):
    k = (id(mesh), kind, key)
    hit = _GCONST.get(k)
    if hit is not None and hit[0] is mesh and hit[1].is_allocated():
        _GCONST[k] = _GCONST.pop(k)  # (most recent last)
        return hit[1]
    while len(_GCONST) >= _GCONST_MAX:
        old = next(iter(_GCONST))
        t = _GCONST.pop(old)[1]
        if t.is_allocated():
            dealloc(t)
    t = make()
    ccl.pin(mesh, t)
    _GCONST[k] = (mesh, t)
    return t


_SCRATCH: dict = {}  # (id(mesh), key spec, j) -> (mesh, a scratch key shard): PB.7C's padding ring buffers


def _scratch_keys(mesh, like: ttnn.Tensor, j: int) -> ttnn.Tensor:
    """PB.7C: a persistent tensor of ``like``'s spec (the ring's local key shard) for padding buffer j (allocated once;
    its content is never read: only one slab of it is gathered)."""
    key = (id(mesh), tuple(like.shape), like.dtype, like.layout, str(like.memory_config()), int(j))
    hit = _SCRATCH.get(key)
    if hit is None or hit[0] is not mesh or not hit[1].is_allocated():
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(like.shape)), like.dtype, like.layout, mesh, like.memory_config()
        )
        t.update_tensor_topology(like.tensor_topology())  # (host metadata: each chip its own shard, as the keys')
        ccl.pin(mesh, t)
        hit = _SCRATCH[key] = (mesh, t)
    return hit[1]


def _chip_u32(mesh, values: tuple) -> ttnn.Tensor:
    """[1, 1, 1, 1] uint32 ROW_MAJOR per chip (chip-linear values)."""

    def make():
        sp, tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        t = torch.tensor([int(v) for v in values], dtype=torch.int32).view(sp, tp, 1, 1)
        return _upload(mesh, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))

    return _gconst(mesh, "u32", tuple(int(v) for v in values), make)


class Call:
    """One ring call of a glued layer's indexer: pass ``p``, segments ``js`` (<= MAX_BUFS, one key capacity), per chip
    its segment (``assign``), its rows [a, b) of it (None: a filler chip, rows not kept) and the roll."""

    def __init__(self, geo: GluedGeo, p: int, js: list, roll: bool = True):
        self.p, self.js = p, list(js)
        self.assign, self.rows = [], []
        for q in range(geo.n):
            e = geo.entry(q, p)
            if e is not None and e[0] in self.js:
                self.assign.append(e[0])
                self.rows.append((e[1], e[2]))
            else:
                self.assign.append(self.js[0])
                self.rows.append(None)
        # (roll False, PB.7C: the ring scored every row in place, lane_ranges; no rolled query copies)
        self.roll = roll and any(r is not None and r[0] > 0 for r in self.rows)
        self.starts = [
            (geo.position(j, q, rw[0]) if rw is not None else int(geo.segs[j].start))
            for q, (j, rw) in enumerate(zip(self.assign, self.rows))
        ]
        self.buf = [self.js.index(j) for j in self.assign]

    def roll_pieces(self, R: int) -> list:
        """Per chip: its rows rolled up by its segment's first local row (identity where none)."""
        out = []
        for rw in self.rows:
            a = rw[0] if rw is not None else 0
            out.append([(a, 0, R - a), (0, R - a, a)] if a else [(0, 0, R)])
        return out

    def phases(self, geo: GluedGeo) -> list:
        """Per chip c: the rows of the call's scores (rolled or not) rolled by c, (i + c) mod R, put each row of its
        segment at the chip row it has when its segment runs alone (row r of a segment at chip row r mod R): the
        top-k splits a chip's last rows over core pairs and a tie's order follows that split (DS41-286 / DS41-342),
        so a row's top-k must run at its alone chip row."""
        R = geo.R
        out = []
        for j, rw in zip(self.assign, self.rows):
            if rw is None:
                out.append(0)
                continue
            a_roll = rw[0] if self.roll else 0
            out.append((a_roll - geo.off[j]) % R)
        return out

    def phase_pieces(self, geo: GluedGeo) -> list:
        R = geo.R
        return [([(0, c, R - c), (R - c, 0, c)] if c else [(0, 0, R)]) for c in self.phases(geo)]

    def keep_pieces(self, geo: GluedGeo, phased: bool) -> list:
        """Per chip: the call's top-k rows of its segment -> the segment's own rows [a, b) (``phased``: the top-k rows
        at the alone chip rows (g - off) mod R, a wrap in two pieces; else at the scores' rows)."""
        R = geo.R
        out = []
        for j, rw in zip(self.assign, self.rows):
            if rw is None:
                out.append([])
                continue
            a, b = rw
            if not phased:
                out.append([((0 if self.roll else a), a, b - a)])
                continue
            t = (a - geo.off[j]) % R
            first = min(b - a, R - t)
            ps = [(t, a, first)]
            if first < b - a:
                ps.append((0, a + first, b - a - first))
            out.append(ps)
        return out


# ---------------------------------------------------------------------------------------------- the chunk
class GluedCall(OneCall):
    """The one-call attention of a glued chunk (tt/packed_device.PackedChunk.oc, PACKED_GLUED_ONECALL=on)."""

    def __init__(self, pk):
        self.pk, self.model = pk, pk.model
        mesh = self.model.mesh
        self.geo = GluedGeo(mesh, pk.desc, pk.segs)
        self.lanes = None
        self.by_lane = {r.lane: r for r in pk.segs}
        self.starts = self.geo.starts
        self.memo: dict = {}
        self.fast = settings.get("PACKED_HOST_FAST") == "on"
        self._tables = None
        self._calls: dict = {}
        # PB.7C attempt 3 (OV-67): every per-chunk table derived on the device from the chunk header (tt/glued_derive)
        self.dv = None
        if dynamic_on():
            from .glued_derive import derive

            self.dv = derive(mesh, self.geo.chunk, int(getattr(pk.desc, "block", 64)), self.geo.align)
            self.dv.begin(pk.desc, getattr(pk, "header", None))
        # PB.HG (PACKED_HIST_OWNED=on): every segment's history to its own chips only, one buffer a source (an idle
        # chip: segment 0's, the lane the sdpa's map gives it); off: every lane's history to every chip
        from . import history_gather

        self.owned = bool(self.geo.align) and history_gather.owned_on()
        self.keys = None
        if self.owned:
            chip_seg = [(e[0] if e is not None else 0) for e in (self.geo.entry(q, 0) for q in range(self.geo.n))]
            history_gather.owned_begin(mesh, self.geo.segs, chip_seg)
            # PB.KR (PACKED_KEYS_OWNED=on): each segment's index keys to its own chips, scored there (tt/owned_keys.py)
            from . import owned_keys

            if owned_keys.on() and dynamic_on():
                self.keys = owned_keys.Keys(mesh, self.geo.segs, chip_seg)
        self.hide = None
        from .gather_hide import Hide, Split, hide_on

        if hide_on():
            split = Split.of(mesh)
            if split is not None:
                self.hide = Hide(self, split)
        STATS["chunks"] += 1
        if STATS["chunks"] == 1:
            from loguru import logger

            logger.info(
                f"PACKED_GLUED_ONECALL=on: one attention call a layer over glued rows {self.geo.off} "
                f"({self.geo.passes} passes)"
            )

    @property
    def tables(self) -> ChipTables:
        if self._tables is None:
            self._tables = ChipTables(self.model.mesh, self.geo.n)
        return self._tables

    def close(self) -> None:
        super().close()
        if self.owned:
            from . import history_gather

            history_gather.owned_end()
        if self._tables is not None:
            self._tables.free()
        if self.dv is not None:
            self.dv.end()

    # ------------------------------------------------------------------ per-descriptor constants
    def _const(self, kind: str, key, make):
        return _gconst(self.model.mesh, kind, (self.geo.key(), key), make)

    def _keep_rows(self) -> ttnn.Tensor:
        g = self.geo
        if self.dv is not None:
            return self.dv.run("keep")[0]
        return self._const(
            "keep", None, lambda: _upload(g.mesh, g.keep_rows(), ttnn.bfloat16, ttnn.TILE_LAYOUT, (0, 1))
        )

    def _ckeep(self, n: int) -> ttnn.Tensor:
        """[1, 1, n, 1] bf16 TILE replicated: the chunk's n natural compressed rows below the used rows."""
        g = self.geo
        if self.dv is not None:
            return self.dv.run("ckeep", (int(n),))[0]

        def make():
            t = torch.zeros(1, 1, n, 1)
            for o, k in zip(g.off, g.rows):
                t[0, 0, o * n // g.chunk : (o + k) * n // g.chunk, 0] = 1.0
            return _upload(g.mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        return self._const("ckeep", int(n), make)

    def _classes(self, i: int) -> tuple:
        return tuple(r.state.window[i].carry_class(r.start) for r in self.pk.segs)

    def _region_index(self, classes: tuple) -> ttnn.Tensor:
        g = self.geo
        if self.dv is not None:  # (the device derives min(start, 128); a masked carry's segment: class 0, its bit)
            mask = 0
            for j, (r, c) in enumerate(zip(self.pk.segs, classes)):
                if int(c) != min(int(r.start), WINDOW):
                    assert int(c) == 0, (j, c, r.start)
                    mask |= 1 << j
            return self.dv.run("region", (), (mask,))[0]
        blocks = [int(r.lane) for r in self.pk.segs]
        return self._const(
            "region",
            tuple(classes),
            lambda: _upload(g.mesh, g.region_lists(classes, blocks), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1)),
        )

    def _srv(self, geo, j: int) -> ttnn.Tensor:
        """Segment j's server-order index row ((d + off) mod S: its rows first), [1, 1, 1, W] uint32 per chip (its SP
        row's), replicated over TP. PB.9G: an encoder replay segment (global_write_start lo > start) writes the rows of
        [lo, cap) as a chunk at lo (TtCompressor._write_from's shifted rows): lo's server order, its rows from lo - start
        on."""
        r = self.geo.segs[j]
        lo = max(int(r.start), int(r.seg.gws))
        k = (lo % geo.chunk) // 32
        off = self.geo.off[j] + (lo - int(r.start))
        if self.dv is not None:
            assert geo.chunk == self.dv.chunk and geo.w == self.dv.W, (geo.chunk, geo.w)
            return self.dv.run("srv", (1,), (j,))[0]

        def make():
            d = geo.server_rows(32 * k)  # [SP, W]
            t = ((d + off) % geo.chunk).to(torch.int32).reshape(geo.sp, 1, 1, geo.w)
            return _upload(self.model.mesh, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, None))

        return _gconst(self.model.mesh, "gsrv", (geo.chunk, int(k), int(off)), make)

    def _gather_sp(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ccl.all_gather(t, 2, SP_AXIS) if self.geo.sp > 1 else t

    def _rows(self, src: ttnn.Tensor, dst: ttnn.Tensor, key, pieces: list, cbytes=None) -> None:
        chip_rows(src, dst, self.tables, key, pieces, cbytes)

    def _like(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(ttnn.Shape(list(t.shape)), t.dtype, t.layout, self.model.mesh, DRAM)

    # ------------------------------------------------------------------ compressor
    def _compress(self, block, x: ttnn.Tensor) -> None:
        pk, m, comp, i = self.pk, self.model, block.compressor, block.layer
        need, los = [], {}
        for r in pk.segs:
            pk.enter(r)
            lo = m.write_from(r.start)  # (DECODER_SKIP=decode: global_write_start; PB.9G: a replay writes from it)
            pk.leave(r)
            if lo < r.run_end:
                assert lo == max(int(r.start), int(r.seg.gws)), (lo, r.seg)
                need.append(r)
                los[id(r)] = lo
        if not need:
            return
        ckv, k = comp(x, self.starts)
        gc, gk = self._gather_sp(ckv), self._gather_sp(k)
        geo = comp.tables.geo
        n = int(gc.shape[-2])
        keep = self._ckeep(n)
        rms = []
        for g in (gc, gk):
            mm_ = ttnn.multiply(g, keep, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(g))
            rm = to_rm(mm_, memory_config=DRAM)
            if rm is not mm_:
                dealloc(mm_)
            rms.append(rm)
        js = {id(r): j for j, r in enumerate(self.geo.segs)}
        plain = [(r, js[id(r)]) for r in need]
        # (PB.7C, PACKED_GLUED_DYNAMIC=on: the one-gather write as PB.7B: its shapes follow the segment count and a
        # cache's slot index only (n <= GLUED_SEGS: a bounded set of programs, built by the first chunks of each count);
        # per-segment writes measured ~+8 ms of host a 5-segment chunk, and padding the gather to GLUED_SEGS would
        # move rows no segment needs, OV-3. DECIDED, OV-58)
        if not (self.fast and len(plain) > 1 and self._glued_write(i, rms, plain, geo)):
            for r, j in plain:
                idx = self._srv(geo, j)
                caches = r.state.compress[i]
                for rm, cache in zip(rms, (caches.ckv, caches.index_k)):
                    assert cache.table is None, "PACKED_ATTN_ONECALL=on: STATIC_TABLES=off (no attention table)"
                    cache.write(rm, los[id(r)], r.run_end, idx=idx)
        for rm in rms:
            dealloc(rm)
        for t, g in ((gc, ckv), (gk, k)):
            if g is not t:
                dealloc(g)
        dealloc(ckv)
        dealloc(k)

    def _glued_write(self, i: int, rms: list, plain: list, geo) -> bool:
        """OneCall._lanes_write with each segment's offset index row (one row gather a cache kind)."""
        from . import cores
        from .compressor import TtServedRows

        caches = [r.state.compress[i] for r, _ in plain]
        kinds = [[c.ckv for c in caches], [c.index_k for c in caches]]
        groups = []
        for cs in kinds:
            blocks = -(-(geo.w // cs[0].ratio) // 32)
            groups.append(cores.max_worker_cores(self.model.mesh) // blocks)
            if not all(c.table is None and cs[0].lanes_ok(c) for c in cs):
                return False
            if cs[0].fp8 and groups[-1] < 1:
                return False
        key = ("gsrv_all", id(geo), tuple(j for _, j in plain))
        idx_all = self.memo.get(key)
        if self.dv is not None:
            assert geo.chunk == self.dv.chunk and geo.w == self.dv.W, (geo.chunk, geo.w)
            idx_all = self.dv.run("srv", (len(plain),), tuple(j for _, j in plain))[0]
        if idx_all is None:
            parts = [self._srv(geo, j) for _, j in plain]
            idx_all = self.memo[key] = ttnn.concat(parts, dim=-1, memory_config=DRAM, sub_core_grids=sub_cores())
        for rm, cs, grp in zip(rms, kinds, groups):
            items = [(c, max(int(r.start), int(r.seg.gws)), r.run_end) for c, (r, _) in zip(cs, plain)]
            TtServedRows.write_lanes(items, rm, idx_all, grp)
        return True

    def _drop(self, key) -> None:
        v = self.memo.pop(key, None)
        if key[0] in ("index", "top", "cands", "srv_all", "gsrv_all", "gidx") and v is not None and v.is_allocated():
            dealloc(v)

    # ------------------------------------------------------------------ indexer
    def _plan(self, kind, caps: list, keys: list) -> list:
        """The layer's ring calls: per pass its segments grouped by (capacity, key), <= MAX_BUFS a call."""
        ck = (kind, tuple(caps), tuple(keys))
        hit = self._calls.get(ck)
        if hit is not None:
            return hit
        g = self.geo
        calls = []
        for p in range(g.passes):
            order = []
            for q in range(g.n):
                e = g.entry(q, p)
                if e is not None and e[0] not in order:
                    order.append(e[0])
            groups: dict = {}
            for j in order:
                groups.setdefault((caps[j], keys[j]), []).append(j)
            for js in groups.values():
                mb = MAX_BUFS[0]
                for c0 in range(0, len(js), mb):
                    calls.append(Call(g, p, js[c0 : c0 + mb]))
        self._calls[ck] = calls
        return calls

    def _index(self, block, x: ttnn.Tensor, qr: ttnn.Tensor, pre=None) -> None:
        from .indexer import Kept, TtCandidateIndexer, TtReindexer, topk_large_indices

        if dynamic_on() and RANGES[0] and dyn("ranges"):
            return self._index_dynamic(block, x, qr, pre)
        ix, i, cfg = block.indexer, block.layer, block.cfg
        assert ix.glue, "PACKED_ATTN_ONECALL=on: INDEXER_GLUE=on"
        assert ix.tp == 1 or ix.local, "PACKED_ATTN_ONECALL=on: INDEXER_TOPK_LOCAL=on"
        if pre is None:
            ix.qr_twin = block.q_a.twin_out if block.q_a.twin else None
            pre = self._query(ix, qr, ix.gates(x))
        w, q, q_kept = pre
        g, mesh = self.geo, self.model.mesh
        segs = g.segs
        caches = [r.state.compress[block.kv_src].index_k for r in segs]
        caps = [ix._capacity(c) for c in caches]
        cand = isinstance(ix, TtCandidateIndexer)
        reidx = isinstance(ix, TtReindexer)
        lens = [min(cp, -(-((r.start + ix.chunk) // ix.ratio) // 32) * 32) for r, cp in zip(segs, caps)]
        nbv = [v // ix.block for v in lens] if cand else [0] * len(segs)
        # (a short candidate pool, fewer blocks than its k: today's padded path, a call per pool length)
        keys = [(v if cand and v < ix.ctopk else None) for v in nbv]
        k_locals = []
        for r, cache in zip(segs, caches):
            ix.share(r.state.memo, block.kv_src)
            shared = ix._memo is not None
            kkey = (r.start, id(cache), cache.version)
            k_local = ix._memo[0].get(("keys", ix._memo[1]), kkey) if shared else None
            if isinstance(k_local, Kept):
                k_local = k_local.t
            assert getattr(cache, "k_served", False), "PACKED_ATTN_ONECALL=on: INDEX_K_SOURCE=served (kept keys)"
            if k_local is None:
                k_local = ix._keys(cache, r.start)
                if shared:
                    ix._memo[0].put(("keys", ix._memo[1]), kkey, Kept(k_local))
            k_locals.append(k_local)
        bias_full = None
        if reidx:
            cands = self.memo.get(("cands", cfg.cand_source))
            assert cands is not None, f"layer {i}: the candidate source {cfg.cand_source} did not run this chunk"
            t0 = caps[0]
            brm = ttnn.scatter(ix._neg[t0], -1, cands, ix.src, memory_config=DRAM, sub_core_grids=sub_cores())
            bias_full = ttnn.to_layout(brm, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(brm))
            dealloc(brm)
            assert len(set(caps)) == 1, "PACKED_GLUED_ONECALL: re-index layers over one key capacity"
        calls = self._plan(("index", type(ix).__name__), caps, keys)
        R = g.R
        top_all, cands_all = None, None
        for ci, c in enumerate(calls):
            t = caps[c.js[0]]
            profiler.signpost("indexer" if ci == 0 else "indexer_pass")  # (PB.7B: the calls after the first)
            if c.roll:
                rk = ("roll", ci, c.p, tuple(c.js))
                qc, wc = self._like(q), self._like(w)
                self._rows(q, qc, rk, c.roll_pieces(R))
                self._rows(w, wc, rk, c.roll_pieces(R))
                bc = None
                if bias_full is not None:
                    bc = self._like(bias_full)
                    self._rows(bias_full, bc, rk, c.roll_pieces(R))
            else:
                qc, wc, bc = q, w, bias_full
            ep = {}
            if cand:
                ep = dict(block_keys=ix.block, block_max_out=ix._bmax[t], block_max_pin=True)
            elif reidx:
                ep = dict(block_keys=ix.block, block_bias=bc)
            clens = [lens[j] for j in c.assign]
            first = segs[c.js[0]]
            logits = ttnn.bringup.ring_indexer_score_dsa(
                qc,
                ix._kbuf[t],
                wc,
                k_locals[c.js[0]],
                ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                cluster_axis=SP_AXIS,
                topology=ix.topology,
                num_links=ix.num_links,
                chunk_start_idx=int(first.start),
                key_compression_ratio=ix.ratio,
                program_config=ix.score_cfg,
                compute_kernel_config=ix.score_ckc,
                kv_len=max(clens),
                seq_subshard_axis=TP_AXIS if ix.tp > 1 else None,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=ix.key_rows,
                natural_query_blocks=True,
                natural_query_rows=0 if ix.key_rows == ix.w_rows else ix.w_rows,
                lane_rows=R,
                lane_starts=[int(v) for v in c.starts],
                lane_kv_lens=[int(v) for v in clens],
                lane_buffer=[int(v) for v in c.buf],
                k_extra=[_kbuf(mesh, ix._kbuf[t], b) for b in range(1, len(c.js))],
                k_local_extra=[k_locals[j] for j in c.js[1:]],
                **ep,
                **_own_extent(),
            )
            if c.roll:
                dealloc(qc)
                dealloc(wc)
                if bc is not None:
                    dealloc(bc)
            # (each row's top-k at its alone chip row: DS41-342)
            phased = any(c.phases(g))
            pk_ = ("phase", ci, c.p, tuple(c.js), c.roll)

            def at_phase(t_in, cols):
                if not phased:
                    return t_in
                profiler.signpost("indexer_phase")  # (PB.7B: the scores to their alone chip rows, DS41-342)
                out_ = self._like(t_in)
                self._rows(t_in, out_, pk_, c.phase_pieces(g), cbytes=cols)
                dealloc(t_in)
                profiler.signpost("indexer" if ci == 0 else "indexer_pass")
                return out_

            cc = None
            if cand:
                from . import cores

                short = keys[c.js[0]]
                bm = ix._bmax[t]
                if short is None:
                    rm = at_phase(cores.typecast(bm, ttnn.bfloat16, memory_config=DRAM), None)
                    cc = topk_large_indices(
                        rm, k=ix.ctopk, valid_length_tensor=_chip_u32(mesh, tuple(nbv[j] for j in c.assign))
                    )
                    dealloc(rm)
                else:  # (TtCandidateIndexer._candidates_glue's padded path, every chip of the call at one pool length)
                    sl = ttnn.slice(
                        bm, [0, 0, 0, 0], [1, 1, ix.sq, short], memory_config=DRAM, sub_core_grids=sub_cores(bm)
                    )
                    rm = cores.typecast(sl, ttnn.bfloat16, memory_config=DRAM)
                    if sl.buffer_address() != bm.buffer_address():
                        dealloc(sl)
                    padded = at_phase(
                        ttnn.pad(rm, [(0, 0), (0, 0), (0, 0), (0, ix.ctopk - short)], value=float("-inf")), None
                    )
                    dealloc(rm)
                    cc = topk_large_indices(padded, k=ix.ctopk)
                    dealloc(padded)
            logits = at_phase(logits, max(clens) * 2)
            vl = _chip_u32(mesh, tuple(clens))
            got = {}

            def topk_step(_, logits=logits, vl=vl):
                got["top"] = topk_large_indices(logits, k=ix.topk, valid_length_tensor=vl)
                return got["top"]

            if self.hide is not None and ci == 0:
                self.hide.topk(i, [topk_step])
            else:
                topk_step(None)
            tc = got["top"]
            dealloc(logits)
            kk = ("keep", ci, c.p, tuple(c.js), c.roll, phased)
            direct = ci == 0 and not c.roll and not phased  # (the call's rows are the chunk's rows already)
            if top_all is None and direct:
                top_all = tc
            else:
                if top_all is None:
                    top_all = self._like(tc)
                self._rows(tc, top_all, kk, c.keep_pieces(g, phased))
                dealloc(tc)
            if cc is not None:
                if cands_all is None and direct:
                    cands_all = cc
                else:
                    if cands_all is None:
                        cands_all = self._like(cc)
                    self._rows(cc, cands_all, kk, c.keep_pieces(g, phased))
                    dealloc(cc)
        STATS["max_calls"] = max(STATS.get("max_calls", 0), len(calls))  # (PB.7C: ring calls this layer)
        if bias_full is not None:
            dealloc(bias_full)
        if not q_kept:
            dealloc(q)
        dealloc(w)
        self._keep(("top", i), top_all)
        if cands_all is not None:
            self._keep(("cands", i), cands_all)

    # ------------------------------------------------------------------ indexer, PB.7C: ONE ring call a layer
    def _ranges(self, lens: list, one_buffer: bool = False) -> list:
        """ring_indexer_score_dsa lane_ranges: per chip [n, (row_lo, row_hi, segment = its key buffer, position of
        row_lo, kv_len) x RANGES_MAX] over its glued rows; rows no segment holds (the tail pad, an empty chip) a filler
        range (buffer 0, position 0, kv_len 32: computed, never kept)."""
        g = self.geo
        out = []
        for q in range(g.n):
            w = []
            last = 0
            for j, a, b in g.chip[q]:
                assert a == last, (q, g.chip[q])
                w.append((a, b, 0 if one_buffer else j, g.position(j, q, a), lens[j]))
                last = b
            if last < g.R:
                w.append((last, g.R, 0, 0, 32))
            assert len(w) <= RANGES_MAX, (q, w)
            flat = [len(w)]
            for r in w:
                flat += [int(v) for v in r]
            out += flat + [0] * (RANGE_STRIDE - len(flat))
        return out

    def _cfg32(self, ix):
        """The ring score's program config at q_chunk 32 (lane_ranges: a range's rows are whole q chunks; one tile
        row a q chunk; the same products and head order as the model's q chunk, so the same bits)."""
        from .indexer import K_CHUNK

        c = ix.__dict__.get("score_cfg32")
        if c is None:
            c = ix.__dict__["score_cfg32"] = ttnn.bringup.IndexerScoreProgramConfig(
                q_chunk_size=32,
                k_chunk_size=K_CHUNK,
                head_group_size=0,
                max_grid_x=ix.score_cfg.max_grid_x,
                max_grid_y=ix.score_cfg.max_grid_y,
            )
        return c

    def _cand_cap(self, ix, nbv: tuple) -> ttnn.Tensor:
        """[SP, TP, 1, ctopk] bf16 TILE per chip: +inf on its first nbv columns, -inf after (min() with it = the
        short candidate pool's -inf pad at a fixed width: TtCandidateIndexer._candidates_glue's padded values)."""
        g = self.geo

        def make():
            c = torch.arange(ix.ctopk).view(1, -1)
            v = torch.tensor([int(x) for x in nbv]).view(-1, 1)
            t = torch.where(c < v, float("inf"), float("-inf")).view(g.sp, g.tp, 1, ix.ctopk)
            return _upload(g.mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT, (0, 1))

        return _gconst(g.mesh, "ccap", (int(ix.ctopk), tuple(int(x) for x in nbv)), make)

    def _index_dynamic(self, block, x: ttnn.Tensor, qr: ttnn.Tensor, pre=None) -> None:
        """PACKED_GLUED_DYNAMIC=on (PB.7C): the layer's indexer scores in ONE ring call (lane_ranges: each chip its
        segments' row ranges, each at its own positions, key buffer and kv_len; runtime args, no program a layout),
        then per pass p (each chip's p-th segment) its top-k at its alone chip rows (DS41-342) with each chip's own
        valid length, copied back to the segment's rows."""
        from . import cores
        from .indexer import (
            BSEL_BLOCK,
            SCORE_BLOCK,
            Kept,
            TtCandidateIndexer,
            TtReindexer,
            topk_large_indices,
            use_block_select,
        )

        ix, i, cfg = block.indexer, block.layer, block.cfg
        assert ix.glue, "PACKED_ATTN_ONECALL=on: INDEXER_GLUE=on"
        assert ix.tp == 1 or ix.local, "PACKED_ATTN_ONECALL=on: INDEXER_TOPK_LOCAL=on"
        if pre is None:
            ix.qr_twin = block.q_a.twin_out if block.q_a.twin else None
            pre = self._query(ix, qr, ix.gates(x))
        w, q, q_kept = pre
        g, mesh = self.geo, self.model.mesh
        segs = g.segs
        caches = [r.state.compress[block.kv_src].index_k for r in segs]
        caps = [ix._capacity(c) for c in caches]
        assert len(set(caps)) == 1, "PACKED_GLUED_DYNAMIC: one key capacity a chunk (DS41-328: the ring's buffers)"
        assert len(segs) <= GLUED_SEGS, len(segs)
        t = caps[0]
        cand = isinstance(ix, TtCandidateIndexer)
        reidx = isinstance(ix, TtReindexer)
        lens = [min(cp, -(-((r.start + ix.chunk) // ix.ratio) // 32) * 32) for r, cp in zip(segs, caps)]
        nbv = [v // ix.block for v in lens] if cand else [0] * len(segs)
        k_locals = []
        for r, cache in zip(segs, caches):
            ix.share(r.state.memo, block.kv_src)
            shared = ix._memo is not None
            kkey = (r.start, id(cache), cache.version)
            k_local = ix._memo[0].get(("keys", ix._memo[1]), kkey) if shared else None
            if isinstance(k_local, Kept):
                k_local = k_local.t
            assert getattr(cache, "k_served", False), "PACKED_ATTN_ONECALL=on: INDEX_K_SOURCE=served (kept keys)"
            if k_local is None:
                k_local = ix._keys(cache, r.start)
                if shared:
                    ix._memo[0].put(("keys", ix._memo[1]), kkey, Kept(k_local))
            k_locals.append(k_local)
        bias_full = None
        if reidx:
            cands = self.memo.get(("cands", cfg.cand_source))
            assert cands is not None, f"layer {i}: the candidate source {cfg.cand_source} did not run this chunk"
            brm = ttnn.scatter(ix._neg[t], -1, cands, ix.src, memory_config=DRAM, sub_core_grids=sub_cores())
            bias_full = ttnn.to_layout(brm, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(brm))
            dealloc(brm)
        keys = getattr(self, "keys", None)
        if keys is not None:  # PB.KR (PACKED_KEYS_OWNED=on): every segment's keys on its own chips, scored there
            kg = keys.finish(block.kv_src)
            assert tuple(kg.shape) == tuple(ix._kbuf[t].shape), (tuple(kg.shape), tuple(ix._kbuf[t].shape))
            kargs = dict(
                k_extra=[], k_local_extra=[], lane_ranges=self._ranges(lens, one_buffer=True), keys_gathered=True
            )
            k_ring, kl0 = kg, k_locals[0]
        else:
            # (a fixed buffer count, GLUED_SEGS: no program a segment count; the extra ones are scratch key shards of
            # the same spec that gather one slab, never read. Distinct tensors: a tensor passed twice would let the
            # op's buffer bindings, resolved once at the program's creation, follow another call's duplicate pattern)
            kl = k_locals + [_scratch_keys(mesh, k_locals[0], b) for b in range(len(k_locals), GLUED_SEGS)]
            kargs = dict(
                k_extra=[_kbuf(mesh, ix._kbuf[t], b) for b in range(1, GLUED_SEGS)],
                k_local_extra=kl[1:],
                lane_ranges=self._ranges(lens),
            )
            k_ring, kl0 = ix._kbuf[t], kl[0]
        ep = {}
        # PB.BM (INDEXER_BLOCK_SELECT): a plain indexer's top-k by block-max pre-selection (the score op writes the
        # block maxes; a host decision on the chunk's largest key count, both paths' programs warmed)
        bsel = use_block_select(ix, max(lens), t, packed=True)
        if cand:
            ep = dict(block_keys=ix.block, block_max_out=ix._bmax[t], block_max_pin=True)
        elif reidx:
            ep = dict(block_keys=ix.block, block_bias=bias_full)
        elif bsel:
            ep = dict(block_keys=SCORE_BLOCK, block_max_out=ix.bsel_buffers(t)[0])
        profiler.signpost("indexer")
        logits = ttnn.bringup.ring_indexer_score_dsa(
            q,
            k_ring,
            w,
            kl0,
            ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            cluster_axis=SP_AXIS,
            topology=ix.topology,
            num_links=ix.num_links,
            chunk_start_idx=0,
            key_compression_ratio=ix.ratio,
            program_config=self._cfg32(ix),
            compute_kernel_config=ix.score_ckc,
            kv_len=max(lens),
            seq_subshard_axis=TP_AXIS if ix.tp > 1 else None,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=ix.key_rows,
            natural_query_blocks=True,
            natural_query_rows=0 if ix.key_rows == ix.w_rows else ix.w_rows,
            **kargs,
            **ep,
        )
        STATS["max_calls"] = max(STATS.get("max_calls", 0), 1)  # (PB.7C: ring calls this layer)
        if bias_full is not None:
            dealloc(bias_full)
        bm32 = None
        if cand:
            bm32 = cores.typecast(ix._bmax[t], ttnn.bfloat16, memory_config=DRAM)
        dv, douts, tabs = self.dv, None, None
        if dv is not None:  # (attempt 3: valid lengths, caps and piece tables of every pass, derived from the header)
            from .glued_derive import slot

            iparams = (int(ix.chunk), int(ix.ratio), int(t), int(ix.block) if cand else BSEL_BLOCK if bsel else 1)
            iparams += (int(ix.ctopk) if cand else 0, bool(cand))
            douts = dv.run("idx", iparams)
            tabs = douts[-1]
            assert g.passes <= GLUED_SEGS, g.passes
        top_all, cands_all = None, None
        for p in range(g.passes):
            js = []
            for qq in range(g.n):
                e = g.entry(qq, p)
                if e is not None and e[0] not in js:
                    js.append(e[0])
            c = Call(g, p, js, roll=False)
            phased = any(c.phases(g))
            pk_ = ("dphase", p)
            if p:
                profiler.signpost("indexer_pass")

            def at_phase(t_in, cols, keep_in=True):
                if not phased:
                    return t_in
                profiler.signpost("indexer_phase")
                out_ = self._like(t_in)
                if dv is not None:
                    _rows_in_rows(t_in, out_, tabs, slot("phase", p), cbytes=cols)
                else:
                    self._rows(t_in, out_, pk_, c.phase_pieces(g), cbytes=cols)
                if not keep_in:
                    dealloc(t_in)
                profiler.signpost("indexer_pass" if p else "indexer")
                return out_

            # (each chip's valid length: its pass-p segment's; a chip without one reads its first tile only)
            clens = [lens[j] if rw is not None else 32 for j, rw in zip(c.assign, c.rows)]
            lg = at_phase(logits, max(clens) * 2)
            vl = douts[p] if dv is not None else _chip_u32(mesh, tuple(clens))
            got = {}

            vlb = None
            if bsel:  # (each chip's valid length in blocks: the derived lens / BSEL_BLOCK)
                vlb = (
                    douts[GLUED_SEGS + p] if dv is not None else _chip_u32(mesh, tuple(v // BSEL_BLOCK for v in clens))
                )

            def topk_step(_, lg=lg, vl=vl, vlb=vlb):
                if vlb is not None:
                    bcols = -(-(max(clens) // BSEL_BLOCK * 2) // 64) * 64
                    got["top"] = ix.block_topk(
                        lg, t, n_tensor=vlb, phase=(lambda b: at_phase(b, bcols)) if phased else None, keys_vl=vl
                    )
                else:
                    profiler.signpost("indexer_topk")  # (PB.BM: the top-k its own section, on and off alike)
                    got["top"] = topk_large_indices(lg, k=ix.topk, valid_length_tensor=vl)
                profiler.signpost("indexer_pass" if p else "indexer")
                return got["top"]

            if self.hide is not None and p == 0:
                self.hide.topk(i, [topk_step])
            else:
                topk_step(None)
            tc = got["top"]
            if lg is not logits:
                dealloc(lg)
            kk = ("dkeep", p, phased)
            direct = p == 0 and not phased
            if top_all is None and direct:
                top_all = tc
            else:
                if top_all is None:
                    top_all = self._like(tc)
                if dv is not None:
                    _rows_in_rows(tc, top_all, tabs, slot("keep", p, phased))
                else:
                    self._rows(tc, top_all, kk, c.keep_pieces(g, phased))
                dealloc(tc)
            if cand:  # the candidate pool: long pools (nbv >= ctopk) by valid length, short ones -inf padded
                real = [rw is not None for rw in c.rows]
                cn = [nbv[j] for j in c.assign]
                parts = []
                if any(r_ and v >= ix.ctopk for r_, v in zip(real, cn)):
                    rm = at_phase(bm32, None)
                    vlc = (
                        douts[GLUED_SEGS + p]
                        if dv is not None
                        else _chip_u32(mesh, tuple(v if r_ and v >= ix.ctopk else ix.ctopk for r_, v in zip(real, cn)))
                    )
                    parts.append((topk_large_indices(rm, k=ix.ctopk, valid_length_tensor=vlc), True))
                    if rm is not bm32:
                        dealloc(rm)
                if any(r_ and v < ix.ctopk for r_, v in zip(real, cn)):
                    # (TtCandidateIndexer._candidates_glue's padded pool at a fixed width: the first ctopk block
                    # maxes, each chip's columns from its nbv on -inf: min() with a per-chip cap row)
                    sl = ttnn.slice(
                        bm32, [0, 0, 0, 0], [1, 1, ix.sq, ix.ctopk], memory_config=DRAM, sub_core_grids=sub_cores(bm32)
                    )
                    tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(sl))
                    if tl is not sl:
                        dealloc(sl)
                    cap = (
                        douts[2 * GLUED_SEGS + p]
                        if dv is not None
                        else self._cand_cap(ix, tuple(v if r_ and v < ix.ctopk else 0 for r_, v in zip(real, cn)))
                    )
                    mn = ttnn.minimum(tl, cap, memory_config=DRAM)
                    dealloc(tl)
                    rmn = to_rm(mn, memory_config=DRAM)
                    if rmn is not mn:
                        dealloc(mn)
                    rm = at_phase(rmn, None, keep_in=False)
                    parts.append((topk_large_indices(rm, k=ix.ctopk), False))
                    dealloc(rm)
                for cc, long_ in parts:
                    pieces = [
                        (ps if (r_ and (v >= ix.ctopk) == long_) else [])
                        for ps, r_, v in zip(c.keep_pieces(g, phased), real, cn)
                    ]
                    if cands_all is None and direct and len(parts) == 1:
                        cands_all = cc
                        continue
                    if cands_all is None:
                        cands_all = self._like(cc)
                    if dv is not None:
                        _rows_in_rows(cc, cands_all, tabs, slot("cand", p, phased, long_))
                    else:
                        self._rows(cc, cands_all, kk + (long_,), pieces)
                    dealloc(cc)
        dealloc(logits)
        if bm32 is not None:
            dealloc(bm32)
        if not q_kept:
            dealloc(q)
        dealloc(w)
        self._keep(("top", i), top_all)
        if cands_all is not None:
            self._keep(("cands", i), cands_all)

    # ------------------------------------------------------------------ window region
    def _gregion(self, att) -> ttnn.Tensor:
        """The layer's glued region [1, 1, 128 + R + 128 GLUED_SEGS, 512] bf16 RM (made once, kept with the layer)."""
        t = att.__dict__.get("glued_region")
        rows = WINDOW + self.geo.R + GLUED_SEGS * WINDOW
        if t is None or not t.is_allocated():
            t = ttnn.zeros(
                [1, 1, rows, 512],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.model.mesh,
                memory_config=DRAM,
            )
            ccl.pin(self.model.mesh, t)
            att.__dict__["glued_region"] = t
        return t

    def _region(self, att, kv: ttnn.Tensor, i: int) -> ttnn.Tensor:
        g, gg = att.geo, self.geo
        assert att.region is not None and att.region.dtype == ttnn.bfloat16, "PACKED_ATTN_ONECALL=on: bf16_tiles"
        assert settings.get("ATTN_FUSE") == "on", "PACKED_ATTN_ONECALL=on: ATTN_FUSE=on"
        assert int(kv.shape[-2]) == g.q_rows == gg.R, (kv.shape, g.q_rows, gg.R)
        segs = gg.segs
        for r in segs:
            assert r.state.window[i].ring is not None, "PACKED_ATTN_ONECALL=on: WINDOW_KV_KEEP=snapshot128 (a ring)"
        region = self._gregion(att)
        om = ttnn.multiply(kv, self._keep_rows(), dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(kv))
        orm = to_rm(om, memory_config=DRAM)
        if om is not kv:
            dealloc(om)
        S = GLUED_SEGS
        pieces, tails, rings, carries = [[] for _ in range(S)], [None] * S, [None] * S, [None] * S
        snap = [snapshot_carry_ok(r.state.window[i], r.start) for r in segs]
        for j, r in enumerate(segs):
            pieces[r.lane] = self._write_own_glued(
                r.state.window[i], orm, r.start, r.run_end, j, ring_carry=not snap[j]
            )
        temps = []
        for j, r in enumerate(segs):
            cache, start, s = r.state.window[i], r.start, r.lane
            rings[s] = cache.ring.t
            if start > 0 and not cache.carry_masked(start):
                assert cache.carry_owner(start) is None, "a ring keeps the carry"
                if snap[j]:  # (the slot's served snapshot, read in place: carry row i at row (start + i) % 128)
                    sn = cache.snap
                    tails[s], carries[s] = sn.tensor, (sn.batch * sn.rows, start % WINDOW)
                    continue
                carry = cache.carry(start)
                tails[s], carries[s] = carry, (0, 0)
                temps.append(carry)
        anchor = next((r.state.window[i].snap.tensor for r in segs if r.state.window[i].snap is not None), None)
        glued_send(self.model.mesh, gg.R)(region, orm, tails, rings, pieces, carries, anchor=anchor)
        dealloc(orm)
        for t in temps:
            dealloc(t)
        for r in segs:
            c = r.state.window[i]
            c.rtail_end = r.run_end if r.run_end == r.start + g.chunk else None
        return region

    def _write_own_glued(self, cache, own_rm, start: int, end: int, j: int, ring_carry: bool = True) -> list:
        """OneCall._write_own for segment j's rows wherever they are (chip q's local rows [a, b))."""
        from . import lean

        gg = self.geo
        assert cache.ring is not None and start % 32 == 0 and start < end, (start, end)
        lo, hi = cache._ring_span(start, end)
        if ring_carry:
            cache._before_ring_write(start)
        else:
            cache._free_pending()
        pieces = []
        pin = True
        if lean.ON[0]:
            pin = cache.pin is not None and max(start, cache.pin_at - WINDOW, 0) < min(
                start + gg.rows[j], cache.pin_at, -(-int(end) // 32) * 32
            )
        for q in range(gg.n):
            ent = [e for e in gg.chip[q] if e[0] == j]
            if not ent:
                continue
            _, a, b = ent[0]
            p0 = gg.position(j, q, a)  # (positions of local rows [a, b))
            x, y = max(lo, p0), min(hi, p0 + b - a)
            if x < y:
                pieces += cache.ring.plan(a + x - p0, x, y - x, end, q)
            if pin:
                cache._pin_put(own_rm, a, p0, b - a, end, q)
        cache._after_ring_write(end)
        return pieces

    # ------------------------------------------------------------------ attention
    def _window(self, block, qr, kv) -> ttnn.Tensor:
        from .attention import SDPA_K_CHUNK
        from .cores import max_grid

        att, i = block.attention, block.layer
        profiler.signpost("attn_region")
        cos, sin = att._rope_rows(self.starts, self.geo.chunk)
        region = self._region(att, kv, i)
        index = self._region_index(self._classes(i))
        profiler.signpost("attention")

        def sdpa(q_rm):  # (TtWindowAttention._sdpa's region path: the fork's BF16 kv)
            o = ttnn.bringup.sparse_sdpa(
                q_rm,
                region,
                index,
                att.d,
                kv_format=ttnn.bringup.SparseKVFormat.BF16,
                scale=1.0,
                k_chunk_size=SDPA_K_CHUNK,
                compute_kernel_config=att.ckc,
                attention_sink=att.sink,
                max_grid=max_grid(),
            )
            dealloc(q_rm)
            return o

        out = self._phase(i, lambda _: att._q(qr, cos, sin), sdpa, lambda o: _out(att, o, cos, sin))
        att.rope.free(cos, sin)
        return out

    def _table(self, block):
        """Every segment's history buffer of the KV source (HISTORY_GATHER=rows, its lane buffer), once a chunk a
        source: ("glued", [buffers], [slab strides])."""
        from . import history_gather

        key = ("table", block.kv_src)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        if self.hide is not None:
            self.hide.flush(block.kv_src)
        att = block.attention
        bufs, strides = [], []
        for r in self.geo.segs:
            ckv = r.state.compress[block.kv_src].ckv
            comp, nb, stride = att._compressed(ckv, r.start)
            assert isinstance(comp, history_gather.Held), "PACKED_ATTN_ONECALL=on: HISTORY_GATHER=rows"
            bufs.append(comp.tensor)
            strides.append(int(stride))
        hit = self.memo[key] = ("glued", bufs, strides)
        return hit

    def _sent(self, like: ttnn.Tensor, topk: int) -> ttnn.Tensor:
        """[1, 1, R, topk] uint32 RM, every value the sentinel (a constant of the mesh)."""

        def make():
            t = torch.full((1, 1, int(like.shape[-2]), topk), SENTINEL, dtype=torch.int64).to(torch.int32)
            return _upload(self.model.mesh, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        return _gconst(self.model.mesh, "sent", (int(like.shape[-2]), int(topk)), make)

    def _sparse(self, block, qr, kv) -> ttnn.Tensor:
        att, i = block.attention, block.layer
        assert not att.static and not att.win8, "PACKED_ATTN_ONECALL=on: STATIC_TABLES=off, bf16_tiles"
        g = self.geo
        profiler.signpost("attn_region")
        cos, sin = att._rope_rows(self.starts, g.chunk)
        region = self._region(att, kv, i)
        profiler.signpost("attn_table")
        _, bufs, strides = self._table(block)
        classes = self._classes(i)
        ikey = ("index", block.idx_src, block.kv_src, classes)
        index = self.memo.get(ikey)
        if index is None:
            profiler.signpost("attn_index")
            comb = self.memo.get(("top", block.idx_src))
            assert comb is not None, f"layer {i}: the index source {block.idx_src} did not run this chunk"
            mine = att._mine(comb)
            ckv = g.segs[0].state.compress[block.kv_src].ckv
            if len(set(strides)) > 1:  # (slot slabs of different sizes: each row its segment's stride)

                def make():
                    t = torch.zeros(g.n, g.R, 1)
                    for q in range(g.n):
                        for j, a, b in g.chip[q]:
                            t[q, a:b, 0] = float(strides[j])
                    return _upload(g.mesh, t.view(g.sp, g.tp, g.R, 1), ttnn.float32, ttnn.TILE_LAYOUT, (0, 1))

                rows = topk_rows_lanes(att, mine, ckv, self._const("stride", tuple(strides), make))
            else:
                rows = att._topk_rows(mine, ckv, strides[0], 0)
            if mine is not comb:
                dealloc(mine)
            win = self._region_index(classes)
            index = ttnn.concat([win, rows], dim=-1, memory_config=DRAM, sub_core_grids=sub_cores())
            dealloc(rows)
            self.memo[ikey] = index
        if g.align and _has_map_runtime():  # PB.9G (chip-aligned lanes): one segment a chip: kv_lanes, the lane map
            # a runtime value (sdpa fork kv_lane_map_runtime): ONE call, no per-row ranges, no program a layout
            loc0 = [(g.entry(q, 0)[0] if g.entry(q, 0) is not None else 0) for q in range(g.n)]
            lbufs = list(bufs) + [bufs[0]] * (GLUED_SEGS - len(bufs))  # (a fixed lane count: no program a count)
            profiler.signpost("attention")

            def sdpa_chips(q_rm):
                o = sdpa_lanes(att, q_rm, lbufs, loc0, index, region, map_runtime=True)
                dealloc(q_rm)
                return o

            out = self._phase(i, lambda _: att._q(qr, cos, sin), sdpa_chips, lambda o: _out(att, o, cos, sin))
            att.rope.free(cos, sin)
            return out
        rows_map = self._lane_rows() if LANE_ROWS[0] else None
        if rows_map is not None:  # (the sdpa fork's kv_lane_rows: every row its segment's history, ONE call)
            loc0 = [(g.entry(q, 0)[0] if g.entry(q, 0) is not None else 0) for q in range(g.n)]
            profiler.signpost("attention")

            # (PB.7C: GLUED_SEGS lane tensors always, the first again as padding: no program a segment count)
            lbufs = list(bufs) + ([bufs[0]] * (GLUED_SEGS - len(bufs)) if dyn("lanes") else [])

            def sdpa_rows(q_rm):
                o = sdpa_lanes(att, q_rm, lbufs, loc0, index, region, lane_rows=rows_map)
                dealloc(q_rm)
                return o

            out = self._phase(i, lambda _: att._q(qr, cos, sin), sdpa_rows, lambda o: _out(att, o, cos, sin))
            att.rope.free(cos, sin)
            return out
        same = len({tuple(b.shape) for b in bufs}) == 1
        plans = []  # per pass: (lane of every chip, its index)
        for p in range(g.passes):
            loc = []
            for q in range(g.n):
                e = g.entry(q, p)
                loc.append(e[0] if e is not None else 0)
            if p == 0 and same:
                plans.append((loc, index))
                continue
            pkey = ("gidx", ikey, p)
            ip = self.memo.get(pkey)
            if ip is None:  # (the pass's rows keep their lists, every other row its window keys only)
                win = self._region_index(classes)
                ip = ttnn.concat(
                    [win, self._sent(win, int(index.shape[-1]) - WINDOW)],
                    dim=-1,
                    memory_config=DRAM,
                    sub_core_grids=sub_cores(),
                )
                self._rows(index, ip, ("pass", p), self._pass_pieces(p))
                self.memo[pkey] = ip
            plans.append((loc, ip))
        profiler.signpost("attention")

        def sdpa(q_rm):
            o = None
            for p, (loc, ip) in enumerate(plans):
                if p:
                    profiler.signpost("attn_pass")  # (PB.7B: a chip's second segment's rows)
                op = sdpa_lanes(att, q_rm, bufs, loc, ip, region)
                if o is None:
                    o = op
                    continue
                self._rows(op, o, ("pass", p), self._pass_pieces(p))
                dealloc(op)
            if len(plans) > 1:
                profiler.signpost("attention")
            dealloc(q_rm)
            return o

        out = self._phase(i, lambda _: att._q(qr, cos, sin), sdpa, lambda o: _out(att, o, cos, sin))
        att.rope.free(cos, sin)
        return out

    def _lane_rows(self) -> list | None:
        """sparse_sdpa kv_lane_rows: per chip [n, (row end, segment) x KV_ROWS_MAX] (the last range to the chip's end;
        a pad-only chip: segment 0), None when a chip holds more than KV_ROWS_MAX segments."""
        g = self.geo
        out = []
        for q in range(g.n):
            ps = g.chip[q]
            if len(ps) > KV_ROWS_MAX:
                return None
            w = [len(ps)] if ps else [1]
            if not ps:
                w += [g.R, 0]
            for k, (j, a, b) in enumerate(ps):
                w += [g.R if k == len(ps) - 1 else b, j]
            w += [0] * (1 + 2 * KV_ROWS_MAX - len(w))
            out += w
        return out

    def _pass_pieces(self, p: int) -> list:
        """Per chip: its pass-p segment's rows (in place)."""
        out = []
        for q in range(self.geo.n):
            e = self.geo.entry(q, p)
            out.append([] if e is None else [(e[1], e[1], e[2] - e[1])])
        return out


STATS = {"chunks": 0, "max_calls": 0}  # glued chunks run through GluedCall since load; the most ring calls a layer
# made (PB.7C, tests/packed/test_glued_walls.py resets it per case)
