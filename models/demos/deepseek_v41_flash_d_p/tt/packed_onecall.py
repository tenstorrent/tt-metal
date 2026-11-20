# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_ATTN_ONECALL=on (PB.3A, owner 2026-11-19; bringup/PACKED_DESIGN.md s5 v2): a packed chunk's attention steps
ONCE a layer over every lane, each chip computing its own lane's segment in place (no lane moves, no 4x compute).

PB.3's v1 (tt/packed_device.py, ONECALL=off) moves each lane's rows to lane 0's chips and runs today's attention
steps once per segment. Every chip computes its rows at positions start + q R + d there, so lane k's rows are right
only after the move. Here every position-dependent quantity becomes a per-chip one instead, read from the chip's lane:

    rope rows        tt/rope.LaneStarts: RopeTable / CompressRope slice each lane's rows at its own start and
                     concatenate them (the natural-order rows of the packed chunk), split over the mesh as today
    q_a, kv, q, wq_b, wo_a, wo_b, the compressor's matmuls, the indexer's gates and query: once (row-local)
    window region   each chip's own rows masked at its lane's valid end (a per-chip keep mask, made once per tuple of
                     lane ends), written into the layer's region; the 128 carry rows: from the previous chip of the
                     lane (the neighbour send, never across a lane boundary), on a lane's first chip from its own
                     copy of the slot's carry (``rtail``, written replicated per segment: a local copy, no send).
                     The ring pieces of every lane's slot go to that lane's chips (LaneSend: one prepared program)
    window index    the region lists of each chip's lane start class (a lane's first chip: its segment's class;
                     every other chip's lists are the class-128 lists, identical on every chip)
    compressor      once; each segment's served writes from the chunk's compressed rows gathered once over SP and
                     rolled so that its lane comes first (the rows v1's move produced, bit for bit)
    indexer score   per segment: one ring call over the segment's slot keys with ttnn.bringup.ring_indexer_score_dsa's
                     natural_query_lane (fork option, PB.3A): the lane's chips score at the segment's positions;
                     the top-k (and the candidate pool) of the segment kept in its slot state as today
    history table   per segment the slot's history gathered into its lane buffer (v1's per-lane buffers), then
                     copied on the lane's chips into one table buffer (LaneCopy): sparse_sdpa reads ONE table, each
                     chip its lane's slot rows
    top-k           each segment's top-k copied on its lane's chips into one buffer (LaneCopy)
    sparse_sdpa     once a layer (window and compressed layers)

Bit-identical to v1 (the loop), segment by segment: a lane's chips run the same ops on the same bytes as lane 0's
chips did in v1 (the oracle, tests/packed/test_packed_oracle.py, with the switch on). A lone segment in lane 0 is
today's chunk (PackedChunk.trivial: this module is not used). Every per-chip table here is a constant of the
descriptor's lane ends / start classes, uploaded once per distinct tuple (``_const``) and kept.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.common.bringup.testing import profiler

from . import ccl, history_gather
from .attention import SENTINEL, WINDOW, _upload
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS
from .cores import sub_cores, to_rm
from .rope import LaneStarts
from .settings import settings

RING_PIECES = 4  # tt/window_send.RING_PIECES
COPY_CORES = (8, 4)  # LaneCopy: a [x, y] rectangle of cores a chip from (0, 0) (data movement only: not a matmul)
COPY_BYTES = 16384  # LaneCopy: L1 bytes of the pages read a NoC barrier


def onecall_on() -> bool:
    """PACKED_ATTN_ONECALL=on (call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED_ATTN_ONECALL") == "on"


# ---------------------------------------------------------------------------------------------- per-chip constants
_CONST: dict = {}  # (id(mesh), kind, key) -> (mesh, tensor): constants of a tuple of lane values, kept for the process
_CONST_MAX = 256


def _const(mesh, kind: str, key, make):
    k = (id(mesh), kind, key)
    hit = _CONST.get(k)
    if hit is not None and hit[0] is mesh and hit[1].is_allocated():
        return hit[1]
    if len(_CONST) >= _CONST_MAX:  # (a bounded set: the oldest goes)
        old = next(iter(_CONST))
        t = _CONST.pop(old)[1]
        if t.is_allocated():
            ttnn.deallocate(t)
    t = make()
    ccl.pin(mesh, t)
    _CONST[k] = (mesh, t)
    return t


class Lanes:
    """The lane geometry of a packed chunk on a mesh: lane k = chips ``chips[k]`` (chip-linear q = r TP + c), each
    chip ``q_rows`` natural chunk rows."""

    def __init__(self, mesh, desc):
        self.mesh = mesh
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        self.n = self.sp * self.tp
        self.lanes, self.L, self.chunk = desc.lanes, desc.lane_rows, desc.chunk
        self.chips = desc.lane_chips(self.sp, self.tp)
        self.lane_of = {q: k for k, cs in enumerate(self.chips) for q in cs}
        self.q_rows = self.chunk // self.n

    def rel(self, q: int) -> int:
        """Chip q's index in its lane."""
        return self.chips[self.lane_of[q]].index(q)

    def keep(self, valid: tuple) -> ttnn.Tensor:
        """[1, 1, q_rows, 1] bf16 TILE per chip: 1.0 on its rows below its lane's valid end (lane-relative), else 0.0
        (tt/attention.WindowTables.mask_chip's mask, each chip its lane's)."""

        def make():
            t = torch.zeros(self.sp, self.tp, self.q_rows, 1)
            for q in range(self.n):
                r, c = divmod(q, self.tp)
                d = self.rel(q) * self.q_rows + torch.arange(self.q_rows)
                t[r, c, :, 0] = (d < int(valid[self.lane_of[q]])).float()
            return _upload(self.mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT, (0, 1))

        return _const(self.mesh, "keep", (self.chunk, self.lanes, tuple(valid)), make)

    def region_index(self, tables, classes: tuple) -> ttnn.Tensor:
        """The window region index lists [1, 1, q_rows, 128] uint32 per chip (tt/attention.WindowTables.region_lists)
        of each chip's start class: a lane's first chip its segment's class, every other chip 128. Class-128 lists are
        the same on every chip (region row i + 1 + j), so all-128 is the load-time table."""
        lane_first = {cs[0]: k for k, cs in enumerate(self.chips)}
        if all(int(c) == WINDOW for c in classes):
            return tables.region_index[WINDOW]

        def make():
            j = torch.arange(WINDOW)
            out = torch.zeros(self.sp, self.tp, self.q_rows, WINDOW, dtype=torch.int32)
            for q in range(self.n):
                r, c = divmod(q, self.tp)
                cls = int(classes[lane_first[q]]) if q in lane_first else WINDOW
                d = (self.rel(q) * self.q_rows + torch.arange(self.q_rows)).unsqueeze(-1)
                d0 = self.rel(q) * self.q_rows
                lo = torch.clamp(d + 1, min=WINDOW - cls)
                n = d + WINDOW - lo + 1
                out[r, c] = torch.where(j < n, lo - d0 + j, torch.full_like(lo + j, SENTINEL)).to(torch.int32)
            return _upload(self.mesh, out, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))

        return _const(self.mesh, "region", (self.chunk, self.lanes, tuple(int(c) for c in classes)), make)

    def lane_rows_keep(self, n: int, valid: tuple) -> ttnn.Tensor:
        """[1, 1, n, 1] bf16 TILE, replicated: the chunk's n natural-order rows (n / lanes a lane), 1.0 on lane k's first
        valid[k] rows, else 0.0 (every lane's rows masked at its segment's end in one multiply)."""

        def make():
            t = torch.zeros(1, 1, n, 1)
            per = n // self.lanes
            for k, v in enumerate(valid):
                t[0, 0, k * per : k * per + int(v), 0] = 1.0
            return _upload(self.mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        return _const(self.mesh, "lane_rows_keep", (int(n), self.lanes, tuple(int(v) for v in valid)), make)

    def srv_lane(self, geo, lane: int, k: int) -> ttnn.Tensor:
        """The server-order index row of start class k (tt/attention.WindowTables.srv_row) with lane ``lane``'s tokens
        first: (d + lane L) mod S, [1, 1, 1, W] uint32 per chip (its SP row's), replicated over TP: a served write of
        that lane's segment reads the chunk's natural-order rows through it (the rows v1's move put first)."""

        def make():
            d = geo.server_rows(32 * k)  # [SP, W]
            t = ((d + lane * self.L) % self.chunk).to(torch.int32).reshape(self.sp, 1, 1, geo.w)
            return _upload(self.mesh, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, None))

        return _const(self.mesh, "srv_lane", (self.chunk, self.lanes, int(lane), int(k)), make)

    def scalar(self, values: tuple) -> ttnn.Tensor:
        """[1, 1, 1, 1] fp32 TILE per chip: its lane's value (a lane without a segment: the given value too)."""

        def make():
            t = torch.zeros(self.sp, self.tp, 1, 1)
            for q in range(self.n):
                r, c = divmod(q, self.tp)
                t[r, c, 0, 0] = float(values[self.lane_of[q]])
            return _upload(self.mesh, t, ttnn.float32, ttnn.TILE_LAYOUT, (0, 1))

        return _const(self.mesh, "scalar", (self.chunk, self.lanes, tuple(float(v) for v in values)), make)

    def u32(self, values: tuple) -> ttnn.Tensor:
        """[1, 1, 1, 1] uint32 ROW_MAJOR per chip: its lane's value (a 1-element device scalar, e.g. topk_large_indices'
        valid_length_tensor)."""

        def make():
            t = torch.zeros(self.sp, self.tp, 1, 1, dtype=torch.int32)
            for q in range(self.n):
                r, c = divmod(q, self.tp)
                t[r, c, 0, 0] = int(values[self.lane_of[q]])
            return _upload(self.mesh, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))

        return _const(self.mesh, "u32", (self.chunk, self.lanes, tuple(int(v) for v in values)), make)


def topk_rows_lanes(att, topk: ttnn.Tensor, ckv_cache, stride_t: ttnn.Tensor) -> ttnn.Tensor:
    """tt/sparse_attention.TtSparseAttention._topk_rows (offset 0) with each chip's own owner stride (``stride_t``, a
    per-chip [1, 1, 1, 1] fp32 scalar: its lane's gather buffer's slab rows; slots whose caches differ in size).
    Integers below 2^24 throughout (exact): the rows of the keys v1 reads in that lane's buffer."""
    from .cores import sub_cores_tiled

    hs = ckv_cache.stripe // ckv_cache.ratio
    n_own = ckv_cache.owners
    f32 = dict(dtype=ttnn.float32, memory_config=DRAM)

    def op(fn, *args):
        out = fn(*args, **f32, sub_core_grids=sub_cores(x))
        for t in args:
            if isinstance(t, ttnn.Tensor) and t is not x and t is not stride_t:
                dealloc(t)
        return out

    def floor_div(v, d):
        y = op(ttnn.multiply, ttnn.add(v, 0.5, **f32, sub_core_grids=sub_cores(v)), 1.0 / d)
        out = ttnn.floor(y, memory_config=DRAM, sub_core_grids=sub_cores(y))
        dealloc(y)
        return out

    tiled = topk.layout != ttnn.TILE_LAYOUT
    tl = ttnn.to_layout(topk, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(topk)) if tiled else topk
    x = ttnn.typecast(
        ttnn.bitcast(tl, ttnn.int32, sub_core_grids=sub_cores(tl)),
        ttnn.float32,
        memory_config=DRAM,
        sub_core_grids=sub_cores(),
    )
    if tiled:
        dealloc(tl)
    s = floor_div(x, hs)
    u = op(ttnn.subtract, x, op(ttnn.multiply, ttnn.clone(s, memory_config=DRAM), float(hs)))
    b = floor_div(s, n_own)
    q = op(ttnn.subtract, s, op(ttnn.multiply, ttnn.clone(b, memory_config=DRAM), float(n_own)))
    row = op(ttnn.add, op(ttnn.add, u, 0.0), op(ttnn.multiply, b, float(hs)))
    row = op(ttnn.add, row, op(ttnn.multiply, q, stride_t))
    valid = ttnn.ge(x, 0.0, dtype=ttnn.float32, sub_core_grids=sub_cores(x))
    out = ttnn.where(valid, row, -1.0, sub_core_grids=sub_cores(valid))
    for t in (x, row, valid):
        dealloc(t)
    i32 = ttnn.typecast(out, ttnn.int32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(out))
    dealloc(out)
    u32 = ttnn.bitcast(i32, ttnn.uint32, sub_core_grids=sub_cores(i32))
    rm = to_rm(u32, memory_config=DRAM)
    if u32 is not i32:
        dealloc(u32)
    dealloc(i32)
    return rm


def snapshot_carry_ok(cache, start: int) -> bool:
    """The window carry [start - 128, start) of a segment is its slot's served snapshot (WINDOW_KV_KEEP=snapshot128):
    the snapshot was written at cap == start (the slot's last chunk ended there: a continuing request) with every
    row of the carry held (no ``snap_missing`` inside it), start >= 128. The snapshot is every chip's copy of the
    ring owners' rows (TtWindowCache.write_snapshot): the bytes the ring carry gathers, read in place by the lane send
    (carry row i = snapshot row (start + i) % 128)."""
    if cache.snap is None or cache.snap_cap != start or start < WINDOW or cache.carry_masked(start):
        return False
    m = cache.snap_missing
    return m is None or m[1] <= start - WINDOW


# ---------------------------------------------------------------------------------------------- LaneCopy
_COPY_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);    // bytes a page (a row)
constexpr uint32_t B = get_compile_time_arg_val(1);       // pages read a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(2);      // cores
constexpr uint32_t NB = get_compile_time_arg_val(3);      // blocks
constexpr uint32_t STRIDE = get_compile_time_arg_val(4);  // source pages from one block's first page to the next's
constexpr uint32_t SLOT = (PAGE + 63) & ~63u;
constexpr uint32_t DSTRIDE = get_compile_time_arg_val(5);  // the destination's
constexpr auto s_args = TensorAccessorArgs<6>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst = get_arg_val<uint32_t>(1);
    const uint32_t k = get_arg_val<uint32_t>(2);  // this core's index of NC
    const uint32_t n = get_arg_val<uint32_t>(3);  // pages a block (a per-run value)
    const auto S = TensorAccessor(s_args, src, PAGE);
    const auto D = TensorAccessor(d_args, dst, PAGE);
    const uint32_t total = NB * n;
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(total) * k) / NC);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(total) * (k + 1)) / NC);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    for (uint32_t i = lo; i < hi; i += B) {
        const uint32_t nb = (hi - i) < B ? (hi - i) : B;
        for (uint32_t t = 0; t < nb; ++t) {
            const uint32_t idx = i + t;
            noc_async_read(S.get_noc_addr((idx / n) * STRIDE + idx % n), buf + t * SLOT, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < nb; ++t) {
            const uint32_t idx = i + t;
            noc_async_write(buf + t * SLOT, D.get_noc_addr((idx / n) * DSTRIDE + idx % n), PAGE);
        }
        noc_async_write_barrier();
    }
}
"""


def _page_bytes(t: ttnn.Tensor) -> int:
    """Bytes of a ROW_MAJOR tensor's page (one last-dim row)."""
    return int(t.shape[-1]) * {ttnn.bfloat16: 2, ttnn.uint8: 1, ttnn.uint32: 4, ttnn.float32: 4}[t.dtype]


class LaneCopy:
    """Pages of ``src`` -> ``dst`` on one lane's chips only (the other chips run nothing): the blocks b < NB of ``n``
    pages, source page b x stride + i -> destination page b x dstride + i (a whole tensor: NB 1). ROW_MAJOR DRAM
    interleaved tensors of the same page size; one prepared program, ``n`` a per-run value. Data movement only."""

    def __init__(self, mesh, chips: list, page: int, blocks: int, stride: int, dstride: int | None = None):
        self.mesh, self.chips, self.page, self.blocks, self.stride = (
            mesh,
            list(chips),
            int(page),
            int(blocks),
            int(stride),
        )
        self.dstride = self.stride if dstride is None else int(dstride)
        self.slot = -(-self.page // 64) * 64
        self.batch = max(1, COPY_BYTES // self.slot)
        x, y = COPY_CORES
        self.cores = [(i, j) for j in range(y) for i in range(x)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))])
        self._pp = None

    def _build(self, src, dst):
        ct = [self.page, self.batch, len(self.cores), self.blocks, self.stride, self.dstride]
        ct += list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
        md = ttnn.MeshProgramDescriptor()
        tp = int(self.mesh.shape[TP_AXIS])
        a = [int(src.buffer_address()), int(dst.buffer_address())]
        for q in self.chips:
            r, c = divmod(q, tp)
            rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
            prog = ttnn.ProgramDescriptor()
            size = self.batch * self.slot + 64
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=self.crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)],
                )
            ]
            rt = ttnn.RuntimeArgs()
            for k, (cx, cy) in enumerate(self.cores):
                rt[cx][cy] = a + [k, 0]
            prog.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_COPY_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=self.crs,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ]
            md[rng] = prog
        return md

    def __call__(self, src: ttnn.Tensor, dst: ttnn.Tensor, n: int) -> None:
        assert _page_bytes(src) == _page_bytes(dst) == self.page, (src.shape, dst.shape, self.page)
        assert (self.blocks - 1) * self.stride + int(n) <= int(src.shape[-2]), (self.blocks, self.stride, n, src.shape)
        assert (self.blocks - 1) * self.dstride + int(n) <= int(dst.shape[-2]), (
            self.blocks,
            self.dstride,
            n,
            dst.shape,
        )
        io = [src, dst]
        if self._pp is None:
            self._pp = ttnn.bringup.prepare_program(self._build(src, dst), io, [(0, 3)])
            want = len(self.chips) * len(self.cores) * 3  # src, dst and the value slot a core
            assert self._pp.patch_count == want, f"LaneCopy: {self._pp.patch_count} patched slots, {want} expected"
        ttnn.bringup.run_prepared(self._pp, io, [int(n)])


_COPIES: dict = {}  # (id(mesh), lane chips, page, blocks, stride, dstride) -> LaneCopy


def lane_copy(mesh, chips, page: int, blocks: int = 1, stride: int = 0, dstride: int | None = None) -> LaneCopy:
    dstride = stride if dstride is None else dstride
    key = (id(mesh), tuple(chips), int(page), int(blocks), int(stride), int(dstride))
    hit = _COPIES.get(key)
    if hit is None or hit.mesh is not mesh:
        hit = _COPIES[key] = LaneCopy(mesh, chips, page, blocks, stride, dstride)
    return hit


# ---------------------------------------------------------------------------------------------- LaneSend
# The window region's fused write per lane (tt/window_send.WindowSend.fused, ATTN_FUSE=on, bf16 regions): the copy
# kernel writes the chip's own rows into its region rows [128, 128 + q_rows) and its lane's ring pieces into its lane's
# slot ring; the send kernel sends its last 128 rows to the next chip of its lane (LANE_SEND), a lane's first chip
# copies its own copy of the slot's carry rows (the replicated ``rtail``) into its region rows [0, 128) (LANE_LOCAL),
# every other chip waits for its neighbour's rows.
_SEND_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#if LANE_SEND
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#endif

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page (a row)
constexpr uint32_t SEG = get_compile_time_arg_val(1);   // bytes a packet (whole 64 B lines)
constexpr uint32_t NSEG = get_compile_time_arg_val(2);  // packets a page
constexpr uint32_t NP = get_compile_time_arg_val(3);    // pages sent (the carry)
constexpr uint32_t B = get_compile_time_arg_val(4);     // pages read a NoC barrier
constexpr uint32_t Q0 = get_compile_time_arg_val(5);    // the first of the chip's own rows sent (its last NP)
constexpr uint32_t LANES = get_compile_time_arg_val(6); // lanes (the carry's page base and rotation a lane)
constexpr uint32_t SLOT = (SEG + 63) & ~63u;
constexpr auto r_args = TensorAccessorArgs<7>();
constexpr auto t_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();

void kernel_main() {
    size_t a = 0;
    const uint32_t region = get_arg_val<uint32_t>(a++);
    const uint32_t tail = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t sem = get_arg_val<uint32_t>(a++);  // the arrival counter (same address on every chip)
    const uint32_t px = get_arg_val<uint32_t>(a++);
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t lane = get_arg_val<uint32_t>(a++);
    const uint32_t tbase = get_arg_val<uint32_t>(a + lane);          // the carry's first page in ``tail``
    const uint32_t trot = get_arg_val<uint32_t>(a + LANES + lane);   // carry row i is page tbase + (i + trot) % NP
    a += 2 * LANES;
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto T = TensorAccessor(t_args, tail, PAGE);
    const auto O = TensorAccessor(o_args, rows, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
#if LANE_SEND
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the lane window send needs a 2D fabric (agent rule 4)"
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
#if !LANE_LOCAL
    (void)tbase;
    (void)trot;
#endif
#if LANE_LOCAL
    // a lane's first chip: its carry rows are its own copy of the slot's carry (rtail, written on every chip)
    for (uint32_t p0 = 0; p0 < NP; p0 += B) {
        const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
        for (uint32_t s = 0; s < NSEG; ++s) {
            const uint32_t off = s * SEG;
            const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(T.get_noc_addr(tbase + (p0 + i + trot) % NP, off), buf + i * SLOT, bytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_write(buf + i * SLOT, R.get_noc_addr(p0 + i, off), bytes);
            }
            noc_async_write_barrier();
        }
    }
#else
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait_min(arrived, 1);
    noc_semaphore_inc(get_noc_addr(sem), 0u - 1u);
    noc_async_atomic_barrier();
#endif
}
"""

_RING_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);   // bytes a row (a page)
constexpr uint32_t B = get_compile_time_arg_val(1);      // rows read a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(2);     // copy cores
constexpr uint32_t RP = get_compile_time_arg_val(3);     // ring pieces a lane
constexpr auto o_args = TensorAccessorArgs<4>();
constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();

// segment 0: the region rows (args 5..7); segment s >= 1: ring piece s - 1 of the chip's lane (args 8 + 3 (lane RP + s - 1))
inline uint32_t seg_arg(uint32_t lane, uint32_t s, uint32_t f) {
    return s == 0 ? get_arg_val<uint32_t>(5 + f) : get_arg_val<uint32_t>(8 + 3 * (lane * RP + s - 1) + f);
}

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t region = get_arg_val<uint32_t>(1);
    const uint32_t ring = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);
    const uint32_t lane = get_arg_val<uint32_t>(4);
    const auto O = TensorAccessor(o_args, rows, PAGE);
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto G = TensorAccessor(g_args, ring, PAGE);
    uint32_t total = 0;
    for (uint32_t s = 0; s <= RP; ++s) {
        total += seg_arg(lane, s, 2);
    }
    const uint32_t lo = total * k / NC;
    const uint32_t hi = total * (k + 1) / NC;
    const uint32_t buf = (get_write_ptr(1) + 63) & ~63u;
    uint32_t g = 0;
    for (uint32_t s = 0; s <= RP; ++s) {
        const uint32_t src0 = seg_arg(lane, s, 0);
        const uint32_t dst0 = seg_arg(lane, s, 1);
        const uint32_t n = seg_arg(lane, s, 2);
        const uint32_t b = lo > g ? lo : g;
        const uint32_t e = hi < g + n ? hi : g + n;
        for (uint32_t r = b; r < e; r += B) {
            const uint32_t nb = (e - r) < B ? (e - r) : B;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(O.get_noc_addr(src0 + r - g + i), buf + i * PAGE, PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t d = dst0 + r - g + i;
                noc_async_write(buf + i * PAGE, s == 0 ? R.get_noc_addr(d) : G.get_noc_addr(d), PAGE);
            }
            noc_async_write_barrier();
        }
        g += n;
    }
}
"""


class LaneSend:
    """The fused window write of a packed chunk (module docstring above ``_SEND_KERNEL``): one prepared program on
    every chip, io [region, rows, tail_0 .. tail_{L-1}, ring_0 .. ring_{L-1}] (a lane without a segment: its own
    scratch tail / ring, never read), the ring pieces of every lane value slots."""

    CORE = (0, 0)
    BATCH_BYTES = 16384
    COPY_BATCH = 8

    def __init__(self, mesh, lanes: Lanes, q_rows: int):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan
        from ttnn.bringup.combine_rf.sparse_combine_l1 import route

        from .window_send import _torus, segments

        self.mesh, self.lanes = mesh, lanes
        R, C = tuple(mesh.shape)
        self.n = R * C
        self.page = 1024  # a bf16 row of 512
        self.carry = WINDOW
        self.q_rows = int(q_rows)
        assert self.q_rows >= WINDOW, f"{q_rows} rows a chip: the carry needs a chip's own rows >= 128"
        self.seg, self.nseg = segments(self.page, int(ttnn.get_tt_fabric_max_payload_size_bytes()))
        slot = -(-self.seg // 64) * 64
        self.slot = slot
        self.batch = max(1, min(self.carry, self.BATCH_BYTES // slot)) if self.nseg == 1 else 1
        chips, _, (wrap_x, wrap_y) = mesh_plan(mesh, 1, _torus(mesh))
        cc = ttnn.CoreCoord(*self.CORE)
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
        self.sem = ttnn.create_global_semaphore(mesh, self.crs, 0, ttnn.BufferType.L1_SMALL)
        sem_addr = int(ttnn.get_global_semaphore_address(self.sem))
        v = mesh.worker_core_from_logical_core(cc)
        node = lambda ch: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))  # noqa: E731
        self._chips = {}  # q: (send, local, args without the io addresses, link or None)
        for q in range(self.n):
            k = lanes.lane_of[q]
            cs = lanes.chips[k]
            j = cs.index(q)
            send, local = j < len(cs) - 1, j == 0
            src = divmod(q, C)
            link, args = None, [0, 0, 0, sem_addr, v.x, v.y, 0, 0]
            if send:
                dst = divmod(cs[j + 1], C)
                hops = route(src, dst, R, C, wrap_x, wrap_y, 0)
                assert hops, (src, dst)
                port = chips[src][hops[0][1]]
                dn = node(dst)
                args = [0, 0, 0, sem_addr, v.x, v.y, int(dn.mesh_id), int(dn.chip_id)]
                link = (node(src), node(port["peer"]), port["links"][0])
            self._chips[q] = (send, local, args, link)
        # (load time: a lane without a segment writes into these; never read)
        self.scratch_tail = [
            ttnn.zeros(
                [1, 1, WINDOW, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM
            )
            for _ in range(lanes.lanes)
        ]
        from .attention import RING

        self.scratch_ring = [
            ttnn.zeros(
                [1, 1, RING, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM
            )
            for _ in range(lanes.lanes)
        ]
        for t in self.scratch_tail + self.scratch_ring:
            ccl.pin(mesh, t)
        self._pps = {}  # (lanes' carry tensors: which share one, their accessors) -> the prepared program

    def _build(self, region, rows, utails, tail_of, rings):
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        L = self.lanes.lanes
        cct = [self.page, self.COPY_BATCH, 1, RING_PIECES] + acc(rows) + acc(region) + acc(rings[0])
        cc = ttnn.CoreCoord(*self.CORE)
        x0, y0 = self.CORE
        md = ttnn.MeshProgramDescriptor()
        R, C = tuple(self.mesh.shape)
        ra, oa = int(region.buffer_address()), int(rows.buffer_address())
        for q in range(self.n):
            r, c = divmod(q, C)
            rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
            send, local, args, link = self._chips[q]
            k = self.lanes.lane_of[q]
            tail = utails[tail_of[k]]
            ct = [self.page, self.seg, self.nseg, self.carry, self.batch, self.q_rows - self.carry, L]
            ct += acc(region) + acc(tail) + acc(rows)
            ta, ga = int(tail.buffer_address()), int(rings[k].buffer_address())
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
            a = [ra, ta, oa] + list(args[3:]) + [k] + [0] * (2 * L)
            if send:
                a += list(ttnn.setup_fabric_connection(self.mesh, *link, p, cc))
            rt = ttnn.RuntimeArgs()
            rt[cc.x][cc.y] = a
            crt = ttnn.RuntimeArgs()
            crt[x0][y0] = [oa, ra, ga, 0, k, 0, WINDOW, self.q_rows] + [0] * (3 * RING_PIECES * L)
            src = f"#define LANE_SEND {int(send)}\n#define LANE_LOCAL {int(local)}\n" + _SEND_KERNEL
            dm = lambda pr, nc: ttnn.DataMovementConfigDescriptor(processor=pr, noc=nc)  # noqa: E731
            p.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=src,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=self.crs,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_RING_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=self.crs,
                    compile_time_args=cct,
                    runtime_args=crt,
                    config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
                ),
            ]
            md[rng] = p
        return md

    def __call__(self, region, rows, tails: list, rings: list, pieces: list, bases=None, rots=None) -> None:
        """region [1, 1, 128 + q_rows, 512] bf16 RM (the layer's), rows [1, 1, q_rows, 512] bf16 RM (each chip its own
        masked rows), per lane: the tensor a lane's first chip copies its carry from (None: the scratch) with the
        carry's first page and rotation (``bases`` / ``rots``, default 0: carry row i = page base + (i + rot) % 128;
        the slot's served snapshot: its slab's first page and start % 128), its slot's ring (None: the scratch) and
        ring pieces [(src row, ring row, rows)]. Lanes may share one carry tensor (the engine's snapshot tensor)."""
        L = self.lanes.lanes
        tails = [t if t is not None else self.scratch_tail[k] for k, t in enumerate(tails)]
        rings = [t if t is not None else self.scratch_ring[k] for k, t in enumerate(rings)]
        bases = [0] * L if bases is None else [int(b) for b in bases]
        rots = [0] * L if rots is None else [int(r) for r in rots]
        assert int(rows.shape[-2]) == self.q_rows and int(region.shape[-2]) == WINDOW + self.q_rows, (
            rows.shape,
            region.shape,
        )
        utails, tail_of = [], []
        for t in tails:  # (one io entry a distinct tensor)
            j = next((u for u, x in enumerate(utails) if x.buffer_address() == t.buffer_address()), None)
            if j is None:
                j = len(utails)
                utails.append(t)
            tail_of.append(j)
        io = [region, rows] + utails + rings
        key = (tuple(tail_of), tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in utails))
        pp = self._pps.get(key)
        if pp is None:
            slots = [(0, 9 + i) for i in range(2 * L)] + [(1, 8 + i) for i in range(3 * RING_PIECES * L)]
            pp = self._pps[key] = ttnn.bringup.prepare_program(
                self._build(region, rows, utails, tail_of, rings), io, slots
            )
            # per chip: the send's region / tail / rows and its carry bases / rotations, the copy core's rows / region /
            # ring and its ring pieces
            want = self.n * (3 + 2 * L + 3 + 3 * RING_PIECES * L)
            assert pp.patch_count == want, f"lane send: {pp.patch_count} patched slots, {want} expected"
        vals = bases + rots
        for k in range(L):
            pk = list(pieces[k]) if k < len(pieces) and pieces[k] else []
            assert len(pk) <= RING_PIECES, pk
            v = []
            for src, dst, n in pk:
                v += [int(src), int(dst), int(n)]
            vals += v + [0] * (3 * RING_PIECES - len(v))
        ttnn.bringup.run_prepared(pp, io, vals)


_SENDS: dict = {}  # (id(mesh), lanes, q_rows) -> LaneSend


def lane_send(mesh, lanes: Lanes) -> LaneSend:
    key = (id(mesh), tuple(tuple(c) for c in lanes.chips), lanes.q_rows)
    hit = _SENDS.get(key)
    if hit is None or hit.mesh is not mesh:
        hit = _SENDS[key] = LaneSend(mesh, lanes, lanes.q_rows)
    return hit


def _has_lane_kv() -> bool:
    """The sdpa fork's kv_lanes option is built (ttnn.bringup.sparse_sdpa's binding names it)."""
    return "kv_lanes" in (getattr(ttnn.bringup.sparse_sdpa, "__doc__", "") or "")


LANE_KV = [_has_lane_kv()]  # (a list: tests may switch it off for the copy path)


def sdpa_lanes(att, q_rm, bufs: list, lane_of_chip: list, index, window) -> ttnn.Tensor:
    """TtSparseAttention._sdpa's ROW_BFP4 + kv_window call (HISTORY_FORMAT=bfp4) with the fork's kv_lanes."""
    from .attention import SDPA_K_CHUNK
    from .cores import max_grid

    assert bufs[0].dtype == ttnn.uint8, "PACKED_ATTN_ONECALL=on: HISTORY_FORMAT=bfp4 (ROW_BFP4 records)"
    return ttnn.bringup.sparse_sdpa(
        q_rm,
        bufs[0],
        index,
        att.d,
        kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4,
        scale=1.0,
        k_chunk_size=SDPA_K_CHUNK,
        compute_kernel_config=att.ckc,
        attention_sink=att.sink,
        kv_window=window,
        max_grid=max_grid(),
        kv_lanes=list(bufs),
        kv_lane_of_chip=[int(v) for v in lane_of_chip],
    )


def _has_one_ring() -> bool:
    """The indexer fork's one-call ring (ring_indexer_score_dsa lane_rows / k_extra) is built."""
    return "lane_rows" in (getattr(ttnn.bringup.ring_indexer_score_dsa, "__doc__", "") or "")


ONE_RING = [_has_one_ring()]  # (a list: tests may switch it off for the per-segment rings)
_KBUFS: dict = {}  # (id(mesh), capacity, j) -> (mesh, lane j's gathered-key buffer, the indexer's kbuf's spec)


def _kbuf(mesh, base: ttnn.Tensor, j: int) -> ttnn.Tensor:
    """Lane buffer j >= 1's ring output of the one-call indexer (the indexer's own ``_kbuf`` is j 0)."""
    key = (id(mesh), tuple(base.shape), base.dtype, j)
    hit = _KBUFS.get(key)
    if hit is not None and hit[0] is mesh and hit[1].is_allocated():
        return hit[1]
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(base.shape)), base.dtype, base.layout, mesh, DRAM)
    ccl.pin(mesh, t)
    _KBUFS[key] = (mesh, t)
    return t


# ---------------------------------------------------------------------------------------------- table buffers
_TABLES: dict = {}  # (id(mesh), shape, dtype) -> (mesh, tensor): one table buffer a history size (every source)
_TOPS: dict = {}  # (id(mesh), shape) -> (mesh, tensor): the combined top-k of a packed chunk


def _buffer(store: dict, mesh, like: ttnn.Tensor, zeros: bool) -> ttnn.Tensor:
    key = (id(mesh), tuple(like.shape), like.dtype)
    hit = store.get(key)
    if hit is not None and hit[0] is mesh and hit[1].is_allocated():
        return hit[1]
    if zeros:
        t = ttnn.zeros(list(like.shape), dtype=like.dtype, layout=like.layout, device=mesh, memory_config=DRAM)
    else:
        t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(like.shape)), like.dtype, like.layout, mesh, DRAM)
    ccl.pin(mesh, t)
    store[key] = (mesh, t)
    return t


# ---------------------------------------------------------------------------------------------- the chunk
STATS = {"chunks": 0}  # packed chunks run through OneCall since load (diagnostic)


class OneCall:
    """The one-call attention of a packed chunk (tt/packed_device.PackedChunk.oc)."""

    def __init__(self, pk):
        self.pk, self.model = pk, pk.model
        self.lanes = Lanes(pk.model.mesh, pk.desc)
        self.by_lane = {r.lane: r for r in pk.segs}
        self.starts = LaneStarts((self.by_lane[k].start if k in self.by_lane else 0) for k in range(self.lanes.lanes))
        self.valid = tuple(
            (self.by_lane[k].run_end - self.by_lane[k].start if k in self.by_lane else 0)
            for k in range(self.lanes.lanes)
        )
        self.memo: dict = {}  # ("table", kv source) -> (T, stride); ("index", idx src, kv src, classes) -> index
        STATS["chunks"] += 1
        if STATS["chunks"] == 1:
            from loguru import logger

            logger.info(f"PACKED_ATTN_ONECALL=on: one attention call a layer over lanes {self.starts}")

    # ------------------------------------------------------------------ helpers
    def close(self) -> None:
        for key in list(self.memo):
            self._drop(key)

    def _drop(self, key) -> None:
        v = self.memo.pop(key, None)
        if key[0] in ("index", "top", "cands") and v is not None and v.is_allocated():
            dealloc(v)

    def _gather_sp(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ccl.all_gather(t, 2, SP_AXIS) if self.lanes.sp > 1 else t

    def _roll(self, g: ttnn.Tensor, lane: int) -> ttnn.Tensor:
        """The natural-order rows ``g`` [1, 1, n, w] TILE with lane ``lane``'s rows first (the order v1's move gave)."""
        n, w = int(g.shape[-2]), int(g.shape[-1])
        off = lane * n // self.lanes.lanes
        if off == 0:
            return g
        head = ttnn.slice(g, [0, 0, off, 0], [1, 1, n, w], memory_config=DRAM, sub_core_grids=sub_cores(g))
        tail = ttnn.slice(g, [0, 0, 0, 0], [1, 1, off, w], memory_config=DRAM, sub_core_grids=sub_cores(g))
        out = ttnn.concat([head, tail], dim=2, memory_config=DRAM, sub_core_grids=sub_cores())
        dealloc(head)
        dealloc(tail)
        return out

    def _classes(self, i: int) -> tuple:
        return tuple(
            (self.by_lane[k].state.window[i].carry_class(self.by_lane[k].start) if k in self.by_lane else WINDOW)
            for k in range(self.lanes.lanes)
        )

    # ------------------------------------------------------------------ the layer
    def attention(self, block, x: ttnn.Tensor) -> ttnn.Tensor:
        """The layer's attention output [1, 1, q_rows, H] for every row of the packed chunk (x: attn_norm, the chips'
        own rows; not consumed)."""
        m, i = self.model, block.layer
        qr, p = block.q_a.with_proj(x)
        profiler.signpost("kv")  # (the step names: the profile's phases, as the single-request path's)
        kv = block.kv.from_proj(p, self.starts)
        dealloc(p)
        if block.compressor is not None:
            profiler.signpost("compressor")
            self._compress(block, x)
        if block.indexer is not None:
            profiler.signpost("indexer")
            self._index(block, x, qr)
        if block.cfg.ratio(i):
            out = self._sparse(block, qr, kv)
        else:
            out = self._window(block, qr, kv)
        dealloc(kv)
        dealloc(qr)
        for r in self.pk.segs:  # (INDEXER_GLUE=on: as TtDS41Block.__call__'s end, every slot state)
            st = r.state
            if st.memo is not None:
                for slot in m.memo_last.get(i, ()):
                    st.memo.drop(slot)
        for slot in m.memo_last.get(i, ()):
            for key in [k for k in self.memo if k[0] == "index" and k[1] == slot[1] and slot[0] == "index"]:
                self._drop(key)
        return out

    def compress_only(self, block, x: ttnn.Tensor) -> None:
        """DECODER_SKIP (kv_only, layer 20): the compressor once, every segment's served writes."""
        self._compress(block, x)

    # ------------------------------------------------------------------ compressor
    def _compress(self, block, x: ttnn.Tensor) -> None:
        pk, m, comp, i = self.pk, self.model, block.compressor, block.layer
        need = []
        for r in pk.segs:
            pk.enter(r)
            lo = m.write_from(r.start)  # DECODER_SKIP=decode (P.51): the segment's global_write_start
            pk.leave(r)
            if lo < r.run_end:  # (an encoder replay segment: its groups are the cached ones)
                need.append((r, lo))
        if not need:
            return
        ckv, k = comp(x, self.starts)
        gc, gk = self._gather_sp(ckv), self._gather_sp(k)
        lanes, geo = self.lanes, comp.tables.geo
        plain = [(r, lo) for r, lo in need if lo <= r.start]
        if plain:  # every lane's rows masked at its end and untilized once; each segment reads its lane's rows
            n = int(gc.shape[-2])
            valid = [0] * lanes.lanes
            for r, _ in plain:
                valid[r.lane] = r.run_end // comp.ratio - r.start // comp.ratio
            keep = lanes.lane_rows_keep(n, tuple(valid))
            rms = []
            for g in (gc, gk):
                m = ttnn.multiply(g, keep, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(g))
                rm = to_rm(m, memory_config=DRAM)
                if rm is not m:
                    dealloc(m)
                rms.append(rm)
            for r, _ in plain:
                idx = lanes.srv_lane(geo, r.lane, (r.start % geo.chunk) // 32)
                caches = r.state.compress[i]
                for rm, cache in zip(rms, (caches.ckv, caches.index_k)):
                    assert cache.table is None, "PACKED_ATTN_ONECALL=on: STATIC_TABLES=off (no attention table)"
                    cache.write(rm, r.start, r.run_end, idx=idx)
            for rm in rms:
                dealloc(rm)
            need = [(r, lo) for r, lo in need if lo > r.start]
        for (
            r,
            lo,
        ) in need:  # (encoder replay segments: their rows shifted to global_write_start, TtCompressor._write_from)
            pre = (self._roll(gc, r.lane), self._roll(gk, r.lane))  # (the rows v1's move put first)
            comp.write(ckv, k, r.start, r.run_end, r.state.compress[i], write_from=lo, gathered=pre)
            for t, g in zip(pre, (gc, gk)):
                if t is not g:
                    dealloc(t)
        for t, g in ((gc, ckv), (gk, k)):
            if g is not t:
                dealloc(g)
        dealloc(ckv)
        dealloc(k)

    # ------------------------------------------------------------------ indexer
    def _index(self, block, x: ttnn.Tensor, qr: ttnn.Tensor) -> None:
        from .indexer import TtCandidateIndexer, TtReindexer

        ix, i, cfg = block.indexer, block.layer, block.cfg
        assert ix.glue, "PACKED_ATTN_ONECALL=on: INDEXER_GLUE=on"
        ix.qr_twin = block.q_a.twin_out if block.q_a.twin else None
        w = ix.gates(x)
        q = ix.query(qr, self.starts)  # (the chips' own rows at their lanes' positions)
        q_kept = False
        if ix.q_bfp4 and q.dtype == ttnn.bfloat4_b:  # ROPE_CHAINS=on: the rope wrote it rounded
            q_kept = ix._q_kept
        elif ix.q_bfp4:
            q4, q_kept = ix._round(q, "query")
            dealloc(q)
            q = q4
        L = self.lanes.L
        if ONE_RING[0] and self._one_ring(block, x, q, w):
            if not q_kept:
                dealloc(q)
            dealloc(w)
            return
        for r in self.pk.segs:
            st, start = r.state, r.start
            cache = st.compress[block.kv_src].index_k
            ix.share(st.memo, block.kv_src)
            lane = (r.lane * L, (r.lane + 1) * L)
            t = ix._capacity(cache)
            if isinstance(ix, TtReindexer):
                cands = self.memo.get(("cands", cfg.cand_source))  # (the one-call ring's pool: every lane's)
                if cands is None:
                    cands = st.get(st.cands, cfg.cand_source, start, "candidate pool")
                brm = ttnn.scatter(ix._neg[t], -1, cands, ix.src, memory_config=DRAM, sub_core_grids=sub_cores())
                bias = ttnn.to_layout(brm, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(brm))
                dealloc(brm)
                logits, kv_len = self._scores(ix, q, w, start, cache, lane, block_keys=ix.block, block_bias=bias)
                dealloc(bias)
                top = ix.topk_of(logits, kv_len)
                dealloc(logits)
            elif isinstance(ix, TtCandidateIndexer):
                ep = dict(block_keys=ix.block, block_max_out=ix._bmax[t], block_max_pin=True)
                logits, kv_len = self._scores(ix, q, w, start, cache, lane, **ep)
                cands = ix._candidates_glue(kv_len, t)
                top = ix.topk_of(logits, kv_len)
                dealloc(logits)
                st.put(st.cands, i, start, cands)
            else:
                logits, kv_len = self._scores(ix, q, w, start, cache, lane)
                top = ix.topk_of(logits, kv_len)
                dealloc(logits)
            st.put(st.topk, i, start, top)
        if not q_kept:
            dealloc(q)
        dealloc(w)

    def _one_ring(self, block, x, q, w) -> bool:
        """The layer's index scores in ONE ring call over every lane (ring_indexer_score_dsa lane_rows: each chip its
        lane's slot keys, start and kv_len; the ring gathers every segment's key shards) and ONE top-k (each chip its
        lane's kv_len, a per-chip valid_length_tensor). The top-k (and the candidate pool) of every lane in one tensor,
        kept for the chunk (``memo`` ("top", layer) / ("cands", layer)). False: not this layer (the slots' key caches
        differ in capacity, or a lane's candidate pool is shorter than its k): the per-segment rings."""
        from .indexer import Kept, TtCandidateIndexer, TtReindexer, topk_large_indices

        ix, i, cfg, lanes = block.indexer, block.layer, block.cfg, self.lanes
        segs = self.pk.segs
        caches = [r.state.compress[block.kv_src].index_k for r in segs]
        caps = [ix._capacity(c) for c in caches]
        cand = isinstance(ix, TtCandidateIndexer)
        # (one key capacity: serving's slots are one size; lanes of different capacities scored wrong keys on the
        # smaller lanes in the fork's lanes ring, DS41-328, tests/packed/test_onecall_sizes.py: per-segment rings)
        if len(segs) > 4 or len(set(caps)) != 1:
            return False
        t = caps[0]
        kv_lens = {r.lane: min(cp, -(-((r.start + ix.chunk) // ix.ratio) // 32) * 32) for r, cp in zip(segs, caps)}
        if cand and min(kv_lens.values()) // ix.block < ix.ctopk:  # (_candidates_glue's padded path: per segment)
            return False
        k_locals = []
        for r, cache in zip(segs, caches):
            st = r.state
            ix.share(st.memo, block.kv_src)
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
        pos = {r.lane: j for j, r in enumerate(segs)}
        first = segs[0]
        L = lanes.lanes
        starts = [(self.by_lane[k].start if k in self.by_lane else first.start) for k in range(L)]
        lens = [kv_lens.get(k, kv_lens[first.lane]) for k in range(L)]
        buf = [pos.get(k, 0) for k in range(L)]
        kbuf = ix._kbuf[t]
        ep = {}
        if cand:
            ep = dict(block_keys=ix.block, block_max_out=ix._bmax[t], block_max_pin=True)
        bias = None
        if isinstance(ix, TtReindexer):
            cands = self.memo.get(("cands", cfg.cand_source))
            if cands is None:
                cands = self._combine(
                    [(r.lane, r.state.get(r.state.cands, cfg.cand_source, r.start, "candidate pool")) for r in segs]
                )
            brm = ttnn.scatter(ix._neg[t], -1, cands, ix.src, memory_config=DRAM, sub_core_grids=sub_cores())
            bias = ttnn.to_layout(brm, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(brm))
            dealloc(brm)
            ep = dict(block_keys=ix.block, block_bias=bias)
        logits = ttnn.bringup.ring_indexer_score_dsa(
            q,
            kbuf,
            w,
            k_locals[0],
            ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            cluster_axis=SP_AXIS,
            topology=ix.topology,
            num_links=ix.num_links,
            chunk_start_idx=first.start,
            key_compression_ratio=ix.ratio,
            program_config=ix.score_cfg,
            compute_kernel_config=ix.score_ckc,
            kv_len=max(lens),
            seq_subshard_axis=TP_AXIS if ix.tp > 1 else None,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=ix.key_rows,
            natural_query_blocks=True,
            natural_query_rows=0 if ix.key_rows == ix.w_rows else ix.w_rows,
            lane_rows=lanes.L,
            lane_starts=[int(v) for v in starts],
            lane_kv_lens=[int(v) for v in lens],
            lane_buffer=[int(v) for v in buf],
            k_extra=[_kbuf(self.model.mesh, ix._kbuf[caps[j]], j) for j in range(1, len(segs))],
            k_local_extra=list(k_locals[1:]),
            **ep,
        )
        if bias is not None:
            dealloc(bias)
        if cand:
            nbv = tuple(v // ix.block for v in lens)
            bm = ix._bmax[t]
            from . import cores

            rm = cores.typecast(bm, ttnn.bfloat16, memory_config=DRAM)
            cands = topk_large_indices(rm, k=ix.ctopk, valid_length_tensor=lanes.u32(nbv))
            dealloc(rm)
            self._keep(("cands", i), cands)
        top = topk_large_indices(logits, k=ix.topk, valid_length_tensor=lanes.u32(tuple(lens)))
        dealloc(logits)
        assert ix.tp == 1 or ix.local, "PACKED_ATTN_ONECALL=on: INDEXER_TOPK_LOCAL=on"
        self._keep(("top", i), top)
        return True

    def _keep(self, key, t: ttnn.Tensor) -> None:
        old = self.memo.pop(key, None)
        if old is not None and old.is_allocated():
            dealloc(old)
        self.memo[key] = t

    def _combine(self, parts: list) -> ttnn.Tensor:
        """[(lane, a per-segment tensor)] -> one buffer of their shape, each lane's chips their segment's (LaneCopy;
        a lane without a segment keeps whatever the buffer held: never read)."""
        like = parts[0][1]
        comb = _buffer(_TOPS, self.model.mesh, like, zeros=True)
        for lane, t in parts:
            assert tuple(t.shape) == tuple(like.shape) and t.dtype == like.dtype, (t.shape, like.shape)
            lane_copy(self.model.mesh, self.lanes.chips[lane], _page_bytes(t))(t, comb, int(t.shape[-2]))
        return comb

    def _scores(self, ix, q, w, start: int, cache, lane: tuple, **epilogue):
        """TtIndexer.scores without the gates / query (made once), for one segment: its slot keys, its start, its
        lane's chips (natural_query_lane)."""
        from .indexer import Kept

        assert start % 32 == 0, start
        t = ix._capacity(cache)
        kbuf = ix._kbuf[t]
        end = start + ix.chunk  # (today's padded window: the same kv_len as v1's per-segment call)
        kv_len = min(t, -(-(end // ix.ratio) // 32) * 32)
        assert kv_len >= ix.topk and start < kv_len * ix.ratio, (start, kv_len, ix.topk)
        shared = ix._memo is not None
        kkey = (start, id(cache), cache.version)
        k_local = ix._memo[0].get(("keys", ix._memo[1]), kkey) if shared else None
        if isinstance(k_local, Kept):
            k_local = k_local.t
        k_kept = (ix.lowbit_host and ix.k_dtype != ttnn.bfloat16) or cache.k8 is not None
        k_kept = k_kept or getattr(cache, "k_served", False)
        if k_local is None:
            k_local = ix._keys(cache, start)
            if shared:
                ix._memo[0].put(("keys", ix._memo[1]), kkey, Kept(k_local) if k_kept else k_local)
        logits = ttnn.bringup.ring_indexer_score_dsa(
            q,
            kbuf,
            w,
            k_local,
            ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            cluster_axis=SP_AXIS,
            topology=ix.topology,
            num_links=ix.num_links,
            chunk_start_idx=start,
            key_compression_ratio=ix.ratio,
            program_config=ix.score_cfg,
            compute_kernel_config=ix.score_ckc,
            kv_len=kv_len,
            seq_subshard_axis=TP_AXIS if ix.tp > 1 else None,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=ix.key_rows,
            natural_query_blocks=True,
            natural_query_rows=0 if ix.key_rows == ix.w_rows else ix.w_rows,
            natural_query_lane_lo=int(lane[0]),
            natural_query_lane_hi=int(lane[1]),
            **epilogue,
        )
        if not (shared or k_kept):
            dealloc(k_local)
        return logits, kv_len

    # ------------------------------------------------------------------ window region
    def _region(self, att, kv: ttnn.Tensor, i: int) -> None:
        """tt/attention.TtWindowAttention._window_region for every lane at once (WINDOW_KV_FORMAT=bf16_tiles,
        WINDOW_KV_KEEP=snapshot128's ring, ATTN_FUSE=on)."""
        g, tables, lanes = att.geo, att.tables, self.lanes
        assert att.region is not None and att.region.dtype == ttnn.bfloat16, "PACKED_ATTN_ONECALL=on: bf16_tiles"
        assert tables.send is not None and settings.get("ATTN_FUSE") == "on", "PACKED_ATTN_ONECALL=on: ATTN_FUSE=on"
        assert int(kv.shape[-2]) == g.q_rows, (kv.shape, g.q_rows)
        segs = self.pk.segs
        # 1. no served write (every slot's cache keeps a ring, WINDOW_KV_KEEP=snapshot128)
        for r in segs:
            assert r.state.window[i].ring is not None, "PACKED_ATTN_ONECALL=on: WINDOW_KV_KEEP=snapshot128 (a ring)"
        # 2. each chip's own rows masked at its lane's end, untilized; the ring pieces of every lane
        om = ttnn.multiply(
            kv, lanes.keep(self.valid), dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(kv)
        )
        orm = to_rm(om, memory_config=DRAM)
        if om is not kv:
            dealloc(om)
        pieces = [[] for _ in range(lanes.lanes)]
        snap = {r.lane: snapshot_carry_ok(r.state.window[i], r.start) for r in segs}
        for r in segs:
            pieces[r.lane] = self._write_own(
                r.state.window[i], orm, r.start, r.run_end, lanes.chips[r.lane], ring_carry=not snap[r.lane]
            )
        # 3. each segment's carry rows (replicated: a lane's first chip copies its own copy in step 4); without a carry
        # read (start 0, a masked carry: rows never attended) the slot's kept tail as it is
        tails = [None] * lanes.lanes
        rings = [None] * lanes.lanes
        bases, rots = [0] * lanes.lanes, [0] * lanes.lanes
        carries = []
        for r in segs:
            cache, start = r.state.window[i], r.start
            tails[r.lane], rings[r.lane] = cache.rtail, cache.ring.t
            if cache.rtail_end != start and start > 0 and not cache.carry_masked(start):
                assert cache.carry_owner(start) is None, "a ring keeps the carry"
                if snap[r.lane]:  # (the slot's served snapshot, read in place: carry row i at row (start + i) % 128)
                    s = cache.snap
                    tails[r.lane], bases[r.lane], rots[r.lane] = s.tensor, s.batch * s.rows, start % WINDOW
                    continue
                carry = cache.carry(start)
                assert carry is not tables.zero_carry, (start, "a carry read returns rows")
                tails[r.lane] = carry
                carries.append(carry)
        # 4. the region rows, the ring pieces and the lanes' neighbour sends in one program
        lane_send(self.model.mesh, lanes)(att.region, orm, tails, rings, pieces, bases, rots)
        dealloc(orm)
        for t in carries:
            dealloc(t)
        for r in segs:
            c = r.state.window[i]
            c.rtail_end = r.run_end if r.run_end == r.start + g.chunk else None

    @staticmethod
    def _write_own(cache, own_rm, start: int, end: int, chips: list, ring_carry: bool = True) -> list:
        """TtWindowCache.write_own(defer=True) for one lane: the segment's rows are on ``chips`` (lane-relative
        offsets j q_rows), each block owned by the chip that holds it. ``ring_carry`` False: the carry is read from the
        served snapshot (``snapshot_carry``), not taken from the ring before this write."""
        from . import lean

        g = cache.geo
        assert cache.ring is not None and start % 32 == 0 and start < end <= start + g.chunk, (start, end)
        lo, hi = cache._ring_span(start, end)
        if ring_carry:
            cache._before_ring_write(start)
        else:
            cache._free_pending()
        q_rows = g.q_rows
        pieces = []
        pin = True
        if lean.ON[0]:
            pin = cache.pin is not None and max(start, cache.pin_at - WINDOW, 0) < min(
                start + len(chips) * q_rows, cache.pin_at, -(-int(end) // 32) * 32
            )
        for j, q in enumerate(chips):
            a, b = max(lo - start, j * q_rows), min(hi - start, (j + 1) * q_rows)
            if a < b:
                pieces += cache.ring.plan(a - j * q_rows, start + a, b - a, end, q)
            if pin:
                cache._pin_put(own_rm, 0, start + j * q_rows, q_rows, end, q)
        cache._after_ring_write(end)
        return pieces

    # ------------------------------------------------------------------ attention
    def _window(self, block, qr, kv) -> ttnn.Tensor:
        att, i = block.attention, block.layer
        profiler.signpost("attn_region")
        cos, sin = att._rope_rows(self.starts, self.lanes.chunk)
        self._region(att, kv, i)
        index = self.lanes.region_index(att.tables, self._classes(i))
        profiler.signpost("attention")
        q_rm = att._q(qr, cos, sin)
        o = att._sdpa(q_rm, att.region, index)
        dealloc(q_rm)
        out = att._out(o, cos, sin)
        dealloc(o)
        att.rope.free(cos, sin)
        return out

    def _table(self, block):
        """The KV source's history of every lane's slot for one sparse_sdpa, once a chunk a source (the first reader
        gathers it): ("lanes", [the lane buffers], lane of every chip, per-lane strides) read by the sdpa fork's
        kv_lanes (each chip its lane's gather buffer, no copy); without that option (``LANE_KV`` False) ("table", T,
        stride): every lane's owner blocks copied on its chips into one buffer (LaneCopy)."""
        key = ("table", block.kv_src)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        att, lanes = block.attention, self.lanes
        parts = []
        for r in self.pk.segs:
            ckv = r.state.compress[block.kv_src].ckv
            comp, nb, stride = att._compressed(ckv, r.start)  # (HISTORY_GATHER=rows: into the lane's buffer)
            assert isinstance(comp, history_gather.Held), "PACKED_ATTN_ONECALL=on: HISTORY_GATHER=rows"
            parts.append((r.lane, comp.tensor, nb * ckv.stripe // ckv.ratio, stride, ckv.owners))
        # (kv_lanes: the lane buffers may differ in rows only, as tests that size each slot's caches to its position
        # make them; serving's slots are one size)
        if LANE_KV[0] and all(tuple(p[1].shape)[-1] == tuple(parts[0][1].shape)[-1] for p in parts):
            bufs = [p[1] for p in parts]
            pos = {p[0]: j for j, p in enumerate(parts)}
            lane_of_chip = [pos.get(lanes.lane_of[q], 0) for q in range(lanes.n)]
            strides = tuple(parts[pos.get(k, 0)][3] for k in range(lanes.lanes))
            hit = self.memo[key] = ("lanes", bufs, lane_of_chip, strides)
            return hit
        # (every lane's owner blocks at the largest slab's stride: the slots' slabs are one size in serving; the
        # tests size each slot's caches to its position. The index rows take that stride: the same keys)
        big = max(parts, key=lambda p: p[3])
        T = _buffer(_TABLES, self.model.mesh, big[1], zeros=True)
        for k in [k for k, v in self.memo.items() if k[0] == "table" and v[0] is T]:
            self.memo.pop(k)  # (another source of this size read the buffer before)
        for lane, src, n, stride, owners in parts:
            lane_copy(self.model.mesh, lanes.chips[lane], _page_bytes(src), owners, stride, big[3])(src, T, n)
        hit = self.memo[key] = ("table", T, big[3])
        return hit

    def _top(self, block):
        """Every segment's top-k of the layer's index source in one buffer (each lane's chips its segment's): the
        one-call ring's own, or the per-segment top-k combined (LaneCopy)."""
        hit = self.memo.get(("top", block.idx_src))
        if hit is not None:
            return hit
        return self._combine(
            [(r.lane, r.state.get(r.state.topk, block.idx_src, r.start, "top-k")) for r in self.pk.segs]
        )

    def _sparse(self, block, qr, kv) -> ttnn.Tensor:
        att, i = block.attention, block.layer
        assert not att.static and not att.win8, "PACKED_ATTN_ONECALL=on: STATIC_TABLES=off, bf16_tiles"
        profiler.signpost("attn_region")
        cos, sin = att._rope_rows(self.starts, self.lanes.chunk)
        self._region(att, kv, i)
        profiler.signpost("attn_table")
        tab = self._table(block)
        classes = self._classes(i)
        ikey = ("index", block.idx_src, block.kv_src, classes)
        index = self.memo.get(ikey)
        if index is None:
            profiler.signpost("attn_index")
            comb = self._top(block)
            mine = att._mine(comb)
            ckv = self.pk.segs[0].state.compress[block.kv_src].ckv
            if tab[0] == "lanes" and len(set(tab[3])) > 1:  # (lane buffers of different strides)
                rows = topk_rows_lanes(att, mine, ckv, self.lanes.scalar(tab[3]))
            else:
                rows = att._topk_rows(mine, ckv, tab[3][0] if tab[0] == "lanes" else tab[2], 0)
            if mine is not comb:
                dealloc(mine)
            win = self.lanes.region_index(att.tables, classes)
            index = ttnn.concat([win, rows], dim=-1, memory_config=DRAM, sub_core_grids=sub_cores())
            dealloc(rows)
            self.memo[ikey] = index
        profiler.signpost("attention")
        q_rm = att._q(qr, cos, sin)
        if tab[0] == "lanes":  # (the sdpa fork's kv_lanes: each chip its lane's gather buffer)
            o = sdpa_lanes(att, q_rm, tab[1], tab[2], index, att.region)
        else:
            o = att._sdpa(q_rm, tab[1], index, window=att.region)
        dealloc(q_rm)
        out = att._out(o, cos, sin)
        dealloc(o)
        att.rope.free(cos, sin)
        return out
