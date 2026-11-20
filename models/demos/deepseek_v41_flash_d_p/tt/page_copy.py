# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_GLUED_DYNAMIC=on (PB.7C): page ranges of an interleaved tensor copied into another in ONE prepared program
whose page offsets and count are per-run values (the same on every chip): no program a position.

``ttnn.slice`` keys its program on the slice's start (a new start: a new program), so every place a glued chunk's
layout or a segment's position picks rows by slicing (a served write's slot / position scalar, one cache's rows of a
multi-segment write) builds programs for new layouts. ``page_copy`` makes the same bytes: whole pages (a ROW_MAJOR
tensor's page is one row; a TILE tensor's one tile) read and written in order, a few cores, data movement only."""

from __future__ import annotations

import ttnn

from .common import DRAM

NCORES = 4  # cores a chip (data movement only; a [NCORES, 1] row from (0, 0))
# PACKED_GLUED_DYNAMIC=on's parts (diagnostic: a test may drop one to find which one changes a result; all on)
DYN_PARTS = {"ranges", "rows", "pick", "pages", "snap", "lanes", "segwrite", "hist"}


# the chunk running now (tt/packed_device.PackedChunk sets it): a fixed-lane chunk ("lanes") keeps PB.7B's programs
# (its layouts repeat by construction: 4 lanes of 2048 rows; the dynamic parts cost it ~3 ms of host and device on
# 4 x 2k, the 67k bar's case); glued and single-segment chunks (and PACKED=off) take the dynamic parts
KIND = ["any"]


def dyn(part: str) -> bool:
    """PACKED_GLUED_DYNAMIC=on and its ``part`` not dropped (DYN_PARTS), outside a fixed-lane chunk."""
    from .settings import settings

    return settings.get("PACKED_GLUED_DYNAMIC") == "on" and part in DYN_PARTS and KIND[0] != "lanes"


BATCH_BYTES = 16384  # L1 bytes of the pages read a NoC barrier

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


class PageCopy:
    """One (source geometry, destination geometry) pair's prepared program (``page_copy``)."""

    def __init__(self, mesh, src: ttnn.Tensor, dst: ttnn.Tensor):
        self.mesh = mesh
        self.page = _page_bytes(src)
        assert self.page == _page_bytes(dst), (src.shape, dst.shape, src.dtype, dst.dtype)
        self.batch = max(1, BATCH_BYTES // (-(-self.page // 64) * 64))
        self._pp = None

    def _build(self, src, dst):
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NCORES - 1, 0))])
        ct = [self.page, self.batch, NCORES]
        for t in (src, dst):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 64 + self.batch * (-(-self.page // 64) * 64)
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
            )
        ]
        rt = ttnn.RuntimeArgs()
        a = [int(src.buffer_address()), int(dst.buffer_address())]
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
        sp, tp = tuple(self.mesh.shape)
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
    not hold its shape: one program for every row count) and its page size."""
    return (tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()), _page_bytes(t))


_BY: dict = {}  # (id(mesh), source geometry, destination geometry) -> (mesh, PageCopy)


def page_copy(
    src: ttnn.Tensor, dst: ttnn.Tensor, s0: int, d0: int, n: int, nb: int = 1, ss: int = 0, ds: int = 0
) -> None:
    """dst's pages [d0 + b ds, d0 + b ds + n) <- src's pages [s0 + b ss, s0 + b ss + n) for blocks b < nb (one page
    size), every chip."""
    mesh = src.device()
    key = (id(mesh), _geo(src), _geo(dst))
    hit = _BY.get(key)
    if hit is None or hit[0] is not mesh:
        hit = _BY[key] = (mesh, PageCopy(mesh, src, dst))
    hit[1](src, dst, s0, d0, n, nb, ss, ds)


def pages_of(src: ttnn.Tensor, s0: int, n: int, shape: list) -> ttnn.Tensor:
    """A new interleaved DRAM tensor of ``shape`` (src's dtype / layout, n pages) holding src's pages [s0, s0 + n)."""
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), src.dtype, src.layout, src.device(), DRAM)
    page_copy(src, out, s0, 0, n)
    return out
