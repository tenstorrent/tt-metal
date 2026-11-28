# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""POS_DYNAMIC=on (MB.2, OWNER 2026-11-20, rule 20: serving never rebuilds a program after warm-up).

``ttnn.slice`` keys its program on the slice's start and end (a new start: a new program), so every place the
single-request path picks rows by a per-chunk value (a chunk start's RoPE rows, a DeviceScalars value, a served slab's
(slot, layer) history rows, the history tail, the server-order row) built programs on fresh traffic (MB.1, DS41-321).
``page_copy`` moves the same bytes in ONE prepared program a (source, destination) geometry whose page offsets and
counts are per-run values (prepared value slots, the same on every chip): whole pages (a ROW_MAJOR tensor's page is one
row, a TILE tensor's one tile) read and written in order on a few cores, data movement only. Ported from the batching
branch (PB.7C a72c8c6918d tt/page_copy.py, PACKED_GLUED_DYNAMIC); ``sp_stride`` (new) gives each mesh row its own
source offset (s0 + r sp_stride: a chip row's natural block of a table's rows, mesh_partition's split, in one copy).

``IdsReorder`` (new): the engine's uint32 input (the server's row order, [SP, 1, W] a mesh row each, TP-replicated) ->
each chip's natural-order ids [1, W] for the embedding, ON THE DEVICE (one SP all_gather of the input + a one-core copy
kernel whose start class is a value slot): positions at or past ``valid`` and ids past the vocabulary become vocab - 1
(the host path's PAD clamp, tt/model._host_emb_ids). No second upload of the ids (the host keeps only its readback for
the engram hash, owner decision).

NC.7 (OWNER 2026-11-27, native C++ port series; brought to model_bringup by MB.NC): PageCopy and IdsReorder run as
the native C++ op ttnn.chunk_program (ttnn/cpp/ttnn/operations/chunk_program: the same kernels, moved verbatim to
device/kernels; the constants of each instance made once into a plan): no prepare_program / run_prepared on the call
path. The native PAGE_COPY has no per-mesh-row offset: an ``sp_stride`` copy (the RoPE / compressed RoPE rows) runs as
``pages_of_pair`` (the GLUED_ROWS kind, cos and sin in one program), or on the Python route when called alone. The
Python prepared route stays reachable for the op test only (``<class>.route = "python"``,
tests/native/test_native_chunk_programs.py)."""

from __future__ import annotations

import ttnn

from .common import DRAM, SP_AXIS, TP_AXIS

NCORES = 4  # cores a chip (data movement only; a [NCORES, 1] row from (0, 0))
# NC.7: ttnn.chunk_program kinds (ttnn/cpp/ttnn/operations/chunk_program/device/chunk_program_device_operation_types.hpp)
K_PAGE_COPY, K_GLUED_ROWS, K_IDS, K_GLUED_IDS, K_DERIVE, K_OWNED, K_PAGE_MULTI = range(7)


def native_plan(mesh, kind: int, consts, groups=(), sems=()):
    """NC.7: the ttnn.chunk_program plan of one program instance (made once; its hash once)."""
    R, C = tuple(mesh.shape)
    return ttnn._ttnn.operations.chunk_program.chunk_program_plan(
        kind, int(R), int(C), [int(x) for x in consts], [[int(x) for x in g] for g in groups], [int(x) for x in sems]
    )


BATCH_BYTES = 16384  # L1 bytes of the pages read a NoC barrier


def on() -> bool:
    """POS_DYNAMIC=on (read per call: a call-time switch)."""
    from .settings import settings

    return settings.get("POS_DYNAMIC") == "on"


def fused() -> bool:
    """NATIVE_FUSIONS=on (NC.9, read per call): the ids reorder in the embedding's reader, page copies into one
    destination in one program (``page_copy_many``)."""
    from .settings import settings

    return settings.get("NATIVE_FUSIONS") == "on"


_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page
constexpr uint32_t B = get_compile_time_arg_val(1);     // pages a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(2);    // cores
constexpr uint32_t SLOT = (PAGE + 63) & ~63u;
constexpr auto s_args = TensorAccessorArgs<3>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst = get_arg_val<uint32_t>(1);
    const uint32_t k = get_arg_val<uint32_t>(2);    // this core's index of NC
    const uint32_t s0 = get_arg_val<uint32_t>(3);   // first source page (per-run value)
    const uint32_t d0 = get_arg_val<uint32_t>(4);   // first destination page (per-run value)
    const uint32_t n = get_arg_val<uint32_t>(5);    // pages a block (per-run value)
    const uint32_t nb = get_arg_val<uint32_t>(6);   // blocks (per-run value)
    const uint32_t ss = get_arg_val<uint32_t>(7);   // source pages between blocks (per-run value)
    const uint32_t ds = get_arg_val<uint32_t>(8);   // destination pages between blocks (per-run value)
    const uint32_t so = get_arg_val<uint32_t>(9);   // this mesh row's source offset (frozen: r x sp_stride)
    const auto S = TensorAccessor(s_args, src, PAGE);
    const auto D = TensorAccessor(d_args, dst, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t total = n * nb;
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(total) * k) / NC);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(total) * (k + 1)) / NC);
    uint32_t g = lo;
    while (g < hi) {
        uint32_t nbat = 0;
        const uint32_t g0 = g;
        for (; nbat < B && g < hi; ++nbat, ++g) {
            const uint32_t blk = g / n, i = g - blk * n;
            noc_async_read(S.get_noc_addr(so + s0 + blk * ss + i), buf + nbat * SLOT, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < nbat; ++t) {
            const uint32_t gg = g0 + t, blk = gg / n, i = gg - blk * n;
            noc_async_write(buf + t * SLOT, D.get_noc_addr(d0 + blk * ds + i), PAGE);
        }
        noc_async_write_barrier();
    }
}
"""


def _page_bytes(t: ttnn.Tensor) -> int:
    if t.layout == ttnn.TILE_LAYOUT:
        return int(ttnn.tile_size(t.dtype))
    return int(t.shape[-1]) * int(t.element_size())


def _core_range():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NCORES - 1, 0))])


class PageCopy:
    """One (source geometry, destination geometry, sp_stride) triple's prepared program (``page_copy``)."""

    route = "native"  # NC.7: "python" = the prepared route (tests/native/test_native_chunk_programs.py only)

    def __init__(self, mesh, src: ttnn.Tensor, dst: ttnn.Tensor, sp_stride: int = 0):
        self.mesh = mesh
        self.page = _page_bytes(src)
        assert self.page == _page_bytes(dst), (src.shape, dst.shape, src.dtype, dst.dtype)
        self.batch = max(1, BATCH_BYTES // (-(-self.page // 64) * 64))
        self.sp_stride = int(sp_stride)
        self._pp = None
        self._nplan = None  # NC.7: the native plan (made at the first call)

    def _build(self, src, dst):
        crs = _core_range()
        ct = [self.page, self.batch, NCORES]
        for t in (src, dst):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 64 + self.batch * (-(-self.page // 64) * 64)
        a = [int(src.buffer_address()), int(dst.buffer_address())]
        sp, tp = tuple(self.mesh.shape)
        md = ttnn.MeshProgramDescriptor()
        rows = range(sp) if self.sp_stride else [None]
        for r in rows:
            prog = ttnn.ProgramDescriptor()
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
                )
            ]
            rt = ttnn.RuntimeArgs()
            so = 0 if r is None else r * self.sp_stride
            for k in range(NCORES):
                rt[k][0] = a + [k, 0, 0, 0, 0, 0, 0, so]
            prog.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ]
            lo, hi = (0, sp - 1) if r is None else (r, r)
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(lo, 0), ttnn.MeshCoordinate(hi, tp - 1))] = prog
        return md, len(rows)

    def __call__(
        self, src: ttnn.Tensor, dst: ttnn.Tensor, s0: int, d0: int, n: int, nb: int = 1, ss: int = 0, ds: int = 0
    ) -> None:
        io = [src, dst]
        if self.route == "native" and not self.sp_stride:  # (MB.NC: sp_stride copies: ``pages_of_pair``)
            plan = getattr(self, "_nplan", None)  # (DS41-353: a held instance from before a reload has none)
            if plan is None:
                plan = self._nplan = native_plan(self.mesh, K_PAGE_COPY, [self.page, self.batch, NCORES])
            ttnn.chunk_program(io, plan, [int(s0), int(d0), int(n), int(nb), int(ss), int(ds)])
            return
        if self._pp is None:
            md, progs = self._build(src, dst)
            self._pp = ttnn.bringup.prepare_program(md, io, [(0, 3 + i) for i in range(6)])
            want = progs * NCORES * 8  # (src, dst and the 6 values a core, every mesh-row program)
            assert self._pp.patch_count == want, f"PageCopy: {self._pp.patch_count} patched slots, {want} expected"
        ttnn.bringup.run_prepared(self._pp, io, [int(s0), int(d0), int(n), int(nb), int(ss), int(ds)])


def _geo(t: ttnn.Tensor) -> tuple:
    """What a program built for ``t`` depends on: its TensorAccessor compile-time args (an interleaved tensor's do
    not hold its shape: one program for every row count; an ND-sharded one's hold its shard geometry), its page size,
    and (FIX.32, DS41-277) its shape when sharded."""
    sharded = t.memory_config().is_sharded()
    return (
        tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()),
        _page_bytes(t),
        tuple(int(x) for x in t.shape) if sharded else None,
    )


_BY: dict = {}  # (id(mesh), source geometry, destination geometry, sp_stride) -> (mesh, PageCopy)


def page_copy(
    src: ttnn.Tensor,
    dst: ttnn.Tensor,
    s0: int,
    d0: int,
    n: int,
    nb: int = 1,
    ss: int = 0,
    ds: int = 0,
    sp_stride: int = 0,
) -> None:
    """dst's pages [d0 + b ds, d0 + b ds + n) <- src's pages [o + s0 + b ss, o + s0 + b ss + n) for blocks b < nb
    (one page size), every chip; o = r x ``sp_stride`` on mesh row r (0 by default)."""
    mesh = src.device()
    key = (id(mesh), _geo(src), _geo(dst), int(sp_stride))
    hit = _BY.get(key)
    if hit is None or hit[0] is not mesh:
        hit = _BY[key] = (mesh, PageCopy(mesh, src, dst, sp_stride))
    if n > 0 and nb > 0:
        hit[1](src, dst, s0, d0, n, nb, ss, ds)


_MANY: dict = {}  # (id(mesh), segments, source geometry, destination geometry) -> (mesh, plan)


def page_copy_many(segs: list) -> None:
    """NC.9: ``page_copy(src, dst, s0, d0, n)`` of every segment (src, dst, s0, d0, n), in ONE ttnn.chunk_program
    (PAGE_MULTI: one page size; every source one accessor geometry, every destination one; destinations disjoint, no
    source a destination). The same bytes as the page copies one after the other; NATIVE_FUSIONS=off: those calls."""
    segs = [s for s in segs if int(s[4]) > 0]
    if not segs:
        return
    if len(segs) == 1 or not fused():
        for src, dst, s0, d0, n in segs:
            page_copy(src, dst, s0, d0, n)
        return
    src0, dst0 = segs[0][0], segs[0][1]
    mesh = src0.device()
    gs, gd = _geo(src0), _geo(dst0)
    if any(_geo(s[0]) != gs or _geo(s[1]) != gd for s in segs[1:]):  # (another geometry: one program can't take it)
        for src, dst, s0, d0, n in segs:
            page_copy(src, dst, s0, d0, n)
        return
    key = (id(mesh), len(segs), gs, gd)
    hit = _MANY.get(key)
    if hit is None or hit[0] is not mesh:
        page = _page_bytes(src0)
        assert page == _page_bytes(dst0), (src0.shape, dst0.shape, src0.dtype, dst0.dtype)
        batch = max(1, BATCH_BYTES // (-(-page // 64) * 64))
        hit = _MANY[key] = (mesh, native_plan(mesh, K_PAGE_MULTI, [page, batch, NCORES, len(segs)]))
    io, vals = [], []
    for src, dst, s0, d0, n in segs:
        io += [src, dst]
        vals += [int(s0), int(d0), int(n)]
    ttnn.chunk_program(io, hit[1], vals)


def pages_of(src: ttnn.Tensor, s0: int, n: int, shape: list, sp_stride: int = 0) -> ttnn.Tensor:
    """A new interleaved DRAM tensor of ``shape`` (src's dtype / layout, n pages) holding src's pages [s0, s0 + n)
    (mesh row r: [s0 + r sp_stride, ..))."""
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), src.dtype, src.layout, src.device(), DRAM)
    page_copy(src, out, s0, 0, n, sp_stride=sp_stride)
    return out


GMAXP = 8  # ttnn.chunk_program GLUED_ROWS: pieces a mesh row (the batching branch's tt/lane_rows.GMAXP)
GWORDS = 1 + 2 * GMAXP  # values a mesh row [pieces, (source unit, units | destination unit << 16) x GMAXP]
_PAIR: dict = {}  # (id(mesh), table geometry) -> (mesh, the native GLUED_ROWS plan of two tables)


def pages_of_pair(a: ttnn.Tensor, b: ttnn.Tensor, s0: int, n: int, shape: list, sp_stride: int):
    """``(pages_of(a, s0, n, shape, sp_stride), pages_of(b, ...))`` (two tables of one geometry: the RoPE cos and sin)
    in ONE native program (MB.NC: ttnn.chunk_program GLUED_ROWS with two pairs, the batching branch's NC.7 cos + sin
    fusion; one piece a mesh row: its pages [s0 + r sp_stride, + n) to page 0). The same bytes as the two page copies
    (tests/native/test_native_chunk_programs.py); ``PageCopy.route = "python"`` keeps the two prepared copies."""
    if PageCopy.route != "native" or _geo(a) != _geo(b):
        return pages_of(a, s0, n, shape, sp_stride), pages_of(b, s0, n, shape, sp_stride)
    mesh = a.device()
    sp = int(mesh.shape[SP_AXIS])
    assert 0 < int(n) < 65536, n
    key = (id(mesh), _geo(a))
    hit = _PAIR.get(key)
    if hit is None or hit[0] is not mesh:
        page = _page_bytes(a)
        batch = max(1, BATCH_BYTES // (-(-page // 64) * 64))
        hit = _PAIR[key] = (mesh, native_plan(mesh, K_GLUED_ROWS, [page, 1, batch, NCORES, GWORDS, 2]))
    vals = []
    for r in range(sp):
        vals += [1, int(s0) + r * int(sp_stride), int(n)] + [0] * (GWORDS - 3)
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), t.dtype, t.layout, mesh, DRAM) for t in (a, b)]
    ttnn.chunk_program([a, outs[0], b, outs[1]], hit[1], vals)
    return outs[0], outs[1]


# ------------------------------------------------------------------------------------- the engine's ids, reordered
_IDS_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t WU = get_compile_time_arg_val(0);    // 32-id units a mesh row (W / 32)
constexpr uint32_t SP = get_compile_time_arg_val(1);    // mesh rows
constexpr uint32_t VMAX = get_compile_time_arg_val(2);  // vocab - 1 (the clamp and the pad id)
constexpr uint32_t UB = 128;                            // bytes a unit (32 uint32)
constexpr auto s_args = TensorAccessorArgs<3>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);  // [SP, 1, W] the server's order (every mesh row's), a page a row
    const uint32_t dst = get_arg_val<uint32_t>(1);  // [1, W] this chip's natural-order ids
    const uint32_t iu = get_arg_val<uint32_t>(2);   // (start mod W) / 32 (per-run value)
    const uint32_t cs = get_arg_val<uint32_t>(3);   // (start / W) mod SP: the mesh row holding the chunk's start
    const uint32_t valid = get_arg_val<uint32_t>(4);  // natural positions below it are ids, the rest pad
    const uint32_t me = get_arg_val<uint32_t>(5);   // this mesh row (frozen)
    const auto S = TensorAccessor(s_args, src, WU * UB);
    const auto D = TensorAccessor(d_args, dst, WU * UB);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t SU = WU * SP;  // units a chunk
    for (uint32_t u = 0; u < WU; ++u) {
        const uint32_t d = me * WU + u;  // the natural unit
        uint32_t row, j;
        if (d + iu < WU) {  // the start row's head (tt/attention.WindowGeometry.server_rows, k == 0)
            row = cs;
            j = d;
        } else if (d + iu >= SU) {  // the start row's tail: the chunk's last iu units
            row = cs;
            j = d + WU - SU;
        } else {
            const uint32_t e = d + iu;
            row = (e / WU + cs) % SP;
            j = e % WU;
        }
        noc_async_read(S.get_noc_addr(row, j * UB), buf + u * UB, UB);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint32_t p0 = me * WU * 32;
    for (uint32_t i = 0; i < WU * 32; ++i) {
        const uint32_t x = v[i];
        v[i] = (p0 + i >= valid || x > VMAX) ? VMAX : x;
    }
    noc_async_write(buf, D.get_noc_addr(0), WU * UB);
    noc_async_write_barrier();
}
"""


class ServerIds:
    """NC.9 (NATIVE_FUSIONS=on): what ``model.reorder_ids`` hands ``model.embed``: the engine's input (not consumed),
    the chunk's start and valid rows; the embedding reads it in server order (``IdsReorder.embed``)."""

    __slots__ = ("inp", "start", "valid")

    def __init__(self, inp, start: int, valid: int):
        self.inp, self.start, self.valid = inp, int(start), int(valid)


class IdsReorder:
    """POS_DYNAMIC=on: the engine's device input -> each chip's natural-order embedding ids, on the device (module
    docstring). One gather output allocated at first use (load-time shaped), one prepared program for every start."""

    route = "native"  # NC.7: "python" = the prepared route (tests/native/test_native_chunk_programs.py only)

    def __init__(self, mesh, chunk: int, vocab: int):
        self.mesh = mesh
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        self.chunk, self.w = int(chunk), int(chunk) // int(mesh.shape[SP_AXIS])
        assert self.w * self.sp == self.chunk and self.w % 32 == 0, (chunk, self.sp)
        self.vmax = int(vocab) - 1
        self._gat = None
        self._pp = None
        self._nplan = None  # NC.7: the native plan (made at the first call)

    def _gather(self, inp: ttnn.Tensor) -> ttnn.Tensor:
        from . import ccl

        if self.sp == 1:
            return inp
        if self._gat is None:
            self._gat = ttnn.allocate_tensor_on_device(
                ttnn.Shape([self.sp, 1, self.w]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM
            )
            ccl._KEEP.add(id(self._gat))
        res = ccl.all_gather(inp, 0, SP_AXIS, out=self._gat)
        if res.buffer_address() != self._gat.buffer_address():  # (the stock fallback: a fresh tensor)
            return res
        return self._gat

    def _build(self, src, dst):
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        wu = self.w // 32
        ct = [wu, self.sp, self.vmax]
        for t in (src, dst):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 64 + wu * 128
        a = [int(src.buffer_address()), int(dst.buffer_address())]
        md = ttnn.MeshProgramDescriptor()
        for r in range(self.sp):
            prog = ttnn.ProgramDescriptor()
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
                )
            ]
            rt = ttnn.RuntimeArgs()
            rt[0][0] = a + [0, 0, 0, r]
            prog.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_IDS_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ]
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, 0), ttnn.MeshCoordinate(r, self.tp - 1))] = prog
        return md

    def embed(self, inp: ttnn.Tensor, start: int, valid: int, weight: ttnn.Tensor, **kw) -> ttnn.Tensor:
        """NC.9 (NATIVE_FUSIONS=on): ``ttnn.bringup.embedding(self(inp, start, valid), weight, **kw)`` in one program
        after the SP all-gather: the embedding's reader reads each natural unit where this class's kernel reads it
        (server_start / server_valid runtime values, clamp_max = vocab - 1). Same bytes, no ids tensor."""
        assert int(inp.shape[-1]) == self.w and inp.dtype == ttnn.uint32, (inp.shape, inp.dtype)
        src = self._gather(inp)
        e = ttnn.bringup.embedding(
            src,
            weight,
            server_start=int(start),
            server_valid=max(0, min(int(valid), self.chunk)),
            clamp_max=self.vmax,
            **kw,
        )
        if src is not self._gat and src is not inp:
            ttnn.deallocate(src)
        return e

    def __call__(self, inp: ttnn.Tensor, start: int, valid: int) -> ttnn.Tensor:
        """inp [SP, 1, W] uint32 (the engine's, a mesh row each; not consumed), the chunk at ``start`` with ``valid``
        computed positions -> [1, W] uint32 ROW_MAJOR DRAM a chip (a new tensor: the caller frees it)."""
        assert int(inp.shape[-1]) == self.w and inp.dtype == ttnn.uint32, (inp.shape, inp.dtype)
        src = self._gather(inp)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.w]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM
        )
        io = [src, out]
        s = int(start)
        iu, cs = (s % self.w) // 32, (s // self.w) % self.sp
        vals = [iu, cs, max(0, min(int(valid), self.chunk))]
        if self.route == "native":
            plan = getattr(self, "_nplan", None)  # (DS41-353: a held instance from before a reload has none)
            if plan is None:
                plan = self._nplan = native_plan(self.mesh, K_IDS, [self.w // 32, self.sp, self.vmax])
            ttnn.chunk_program(io, plan, vals)
        else:
            if self._pp is None:
                self._pp = ttnn.bringup.prepare_program(self._build(src, out), io, [(0, 2), (0, 3), (0, 4)])
                want = self.sp * 5  # (src, dst and the 3 values, a core a mesh row)
                assert (
                    self._pp.patch_count == want
                ), f"IdsReorder: {self._pp.patch_count} patched slots, {want} expected"
            ttnn.bringup.run_prepared(self._pp, io, vals)
        if src is not self._gat and src is not inp:
            ttnn.deallocate(src)
        return out
