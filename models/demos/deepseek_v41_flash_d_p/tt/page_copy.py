# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""POS_DYNAMIC=on (MB.2, OWNER 2026-11-20, rule 20: serving never rebuilds a program after warm-up).

``ttnn.slice`` keys its program on the slice's start and end (a new start: a new program), so every place the
single-request path picks rows by a per-chunk value (a chunk start's RoPE rows, a DeviceScalars value, a served slab's
(slot, layer) history rows, the history tail, the server-order row) built programs on fresh traffic (MB.1, DS41-347).
``page_copy`` moves the same bytes in ONE prepared program a (source, destination) geometry whose page offsets and
counts are per-run values (prepared value slots, the same on every chip): whole pages (a ROW_MAJOR tensor's page is one
row, a TILE tensor's one tile) read and written in order on a few cores, data movement only (PB.7C's, ported to
model_bringup by MB.2; MB.2's ``sp_stride`` option was dropped in PB.MB: its only users, the RoPE / compressed RoPE
rows, take PB.7C's GluedRows, one start being one glued piece).
One mechanism for both paths (PB.MB): the batching branch's PACKED_GLUED_DYNAMIC=on parts (``dyn``: a glued chunk's
layout-derived rows, served write slots / positions) and POS_DYNAMIC=on (``on``: the single-request path's per-chunk
starts) both go through this one ``page_copy``.

``IdsReorder`` (new): the engine's uint32 input (the server's row order, [SP, 1, W] a mesh row each, TP-replicated) ->
each chip's natural-order ids [1, W] for the embedding, ON THE DEVICE (one SP all_gather of the input + a one-core copy
kernel whose start class is a value slot): positions at or past ``valid`` and ids past the vocabulary become vocab - 1
(the host path's PAD clamp, tt/model._host_emb_ids). No second upload of the ids (the host keeps only its readback for
the engram hash, owner decision)."""

from __future__ import annotations

import ttnn

from .common import DRAM, SP_AXIS, TP_AXIS

NCORES = 4  # cores a chip (data movement only; a [NCORES, 1] row from (0, 0))
BATCH_BYTES = 16384  # L1 bytes of the pages read a NoC barrier
# the runtime-value parts (diagnostic: a test may drop one to find which one changes a result; all on).
# PACKED_GLUED_DYNAMIC=on (PB.7C) turns on GLUED_PARTS, POS_DYNAMIC=on (MB.2) POS_PARTS: one predicate (``dyn``) and
# one implementation a site for both (PB.MB): "rows" RoPE / compressed RoPE rows by start (GluedRows), "pick"
# DeviceScalars values, "pages" the server-order row and the window carry, "snap" the snapshot send's block pattern,
# "hist" the history pack / tail / gather (fixed row counts), "slab" a served slab's (slot, layer) rows, "replay" the
# compressor replay's rotation, "decode" the index-key decode's grid; "ranges", "lanes", "segwrite" are glued-only
GLUED_PARTS = {"ranges", "rows", "pick", "pages", "snap", "lanes", "segwrite", "hist"}
POS_PARTS = {"rows", "pick", "pages", "snap", "hist", "slab", "replay", "decode"}
DYN_PARTS = GLUED_PARTS | POS_PARTS


# the chunk running now (tt/packed_device.PackedChunk sets it): a fixed-lane chunk ("lanes") keeps PB.7B's programs
# (its layouts repeat by construction: 4 lanes of 2048 rows; the dynamic parts cost it ~3 ms of host and device on
# 4 x 2k, the 67k bar's case); glued and single-segment chunks (and PACKED=off) take the dynamic parts
KIND = ["any"]


def on() -> bool:
    """POS_DYNAMIC=on (read per call: a call-time switch)."""
    from .settings import settings

    return settings.get("POS_DYNAMIC") == "on"


def dyn(part: str) -> bool:
    """``part`` as runtime values: PACKED_GLUED_DYNAMIC=on (GLUED_PARTS) or POS_DYNAMIC=on (POS_PARTS), the part not
    dropped (DYN_PARTS), outside a fixed-lane chunk (fixed lanes keep PB.7B's programs: out of PB.MB's scope)."""
    from .settings import settings

    if part not in DYN_PARTS or KIND[0] == "lanes":
        return False
    return (part in POS_PARTS and on()) or (part in GLUED_PARTS and settings.get("PACKED_GLUED_DYNAMIC") == "on")


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
            noc_async_read(S.get_noc_addr(s0 + blk * ss + i), buf + nbat * SLOT, PAGE);
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
    """One (source geometry, destination geometry) pair's prepared program (``page_copy``)."""

    def __init__(self, mesh, src: ttnn.Tensor, dst: ttnn.Tensor):
        self.mesh = mesh
        self.page = _page_bytes(src)
        assert self.page == _page_bytes(dst), (src.shape, dst.shape, src.dtype, dst.dtype)
        self.batch = max(1, BATCH_BYTES // (-(-self.page // 64) * 64))
        self._pp = None

    def _build(self, src, dst):
        crs = _core_range()
        ct = [self.page, self.batch, NCORES]
        for t in (src, dst):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 64 + self.batch * (-(-self.page // 64) * 64)
        a = [int(src.buffer_address()), int(dst.buffer_address())]
        sp, tp = tuple(self.mesh.shape)
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
            )
        ]
        rt = ttnn.RuntimeArgs()
        for k in range(NCORES):
            rt[k][0] = a + [k, 0, 0, 0, 0, 0, 0]
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
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(sp - 1, tp - 1))] = prog
        return md

    def __call__(
        self, src: ttnn.Tensor, dst: ttnn.Tensor, s0: int, d0: int, n: int, nb: int = 1, ss: int = 0, ds: int = 0
    ) -> None:
        io = [src, dst]
        if self._pp is None:
            self._pp = ttnn.bringup.prepare_program(self._build(src, dst), io, [(0, 3 + i) for i in range(6)])
            want = NCORES * 8  # (src, dst and the 6 values a core)
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


_BY: dict = {}  # (id(mesh), source geometry, destination geometry) -> (mesh, PageCopy)


def page_copy(
    src: ttnn.Tensor,
    dst: ttnn.Tensor,
    s0: int,
    d0: int,
    n: int,
    nb: int = 1,
    ss: int = 0,
    ds: int = 0,
) -> None:
    """dst's pages [d0 + b ds, d0 + b ds + n) <- src's pages [s0 + b ss, s0 + b ss + n) for blocks b < nb (one page
    size), every chip (nothing when n or nb is 0)."""
    mesh = src.device()
    key = (id(mesh), _geo(src), _geo(dst))
    hit = _BY.get(key)
    if hit is None or hit[0] is not mesh:
        hit = _BY[key] = (mesh, PageCopy(mesh, src, dst))
    if n > 0 and nb > 0:
        hit[1](src, dst, s0, d0, n, nb, ss, ds)


def pages_of(src: ttnn.Tensor, s0: int, n: int, shape: list) -> ttnn.Tensor:
    """A new interleaved DRAM tensor of ``shape`` (src's dtype / layout, n pages) holding src's pages [s0, s0 + n)."""
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), src.dtype, src.layout, src.device(), DRAM)
    page_copy(src, out, s0, 0, n)
    return out


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


class IdsReorder:
    """POS_DYNAMIC=on: the engine's device input -> each chip's natural-order embedding ids, on the device (module
    docstring). One gather output allocated at first use (load-time shaped), one prepared program for every start."""

    def __init__(self, mesh, chunk: int, vocab: int):
        self.mesh = mesh
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        self.chunk, self.w = int(chunk), int(chunk) // int(mesh.shape[SP_AXIS])
        assert self.w * self.sp == self.chunk and self.w % 32 == 0, (chunk, self.sp)
        self.vmax = int(vocab) - 1
        self._gat = None
        self._pp = None

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

    def __call__(self, inp: ttnn.Tensor, start: int, valid: int) -> ttnn.Tensor:
        """inp [SP, 1, W] uint32 (the engine's, a mesh row each; not consumed), the chunk at ``start`` with ``valid``
        computed positions -> [1, W] uint32 ROW_MAJOR DRAM a chip (a new tensor: the caller frees it)."""
        assert int(inp.shape[-1]) == self.w and inp.dtype == ttnn.uint32, (inp.shape, inp.dtype)
        src = self._gather(inp)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.w]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM
        )
        io = [src, out]
        if self._pp is None:
            self._pp = ttnn.bringup.prepare_program(self._build(src, out), io, [(0, 2), (0, 3), (0, 4)])
            want = self.sp * 5  # (src, dst and the 3 values, a core a mesh row)
            assert self._pp.patch_count == want, f"IdsReorder: {self._pp.patch_count} patched slots, {want} expected"
        s = int(start)
        iu, cs = (s % self.w) // 32, (s // self.w) % self.sp
        ttnn.bringup.run_prepared(self._pp, io, [iu, cs, max(0, min(int(valid), self.chunk))])
        if src is not self._gat and src is not inp:
            ttnn.deallocate(src)
        return out
