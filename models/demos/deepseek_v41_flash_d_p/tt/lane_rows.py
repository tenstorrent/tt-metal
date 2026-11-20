# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_HOST_FAST=on (PB.3K): a packed chunk's rope rows of every lane, each chip its own rows, in ONE program.

PB.3A's path (tt/rope.lane_slices + mesh_partition) builds the natural-order rows of the whole packed chunk (each lane
``lr`` rows of the table from its own start: 4 slices and a concat) on every chip and then keeps each chip's SP-row
block (mesh_partition): 6 launches a table (cos, sin) a rope kind, ~0.4 ms of host a table on the 8x4 (PB.3K census).

``LaneRows`` writes, on every chip, exactly the rows mesh_partition left it: SP row r holds the natural rows
[r R, (r + 1) R) of the lane order (R = n / SP), i.e. the pieces of the lanes overlapping that block (8x4: one piece, its
lane's start plus its offset in the lane; 2x2 / 2x4: two lanes an SP row). The pieces of each SP row are a constant of
the geometry (a per-chip word table, made once, sharded over SP); the 4 lane starts are per-run values (prepared
program value slots, the same on every chip). Data movement only: page copies (a table row for a ROW_MAJOR table, a
tile for a TILE one: starts and blocks in whole tile rows), the same bytes as the slices. One program for the mesh, a
few cores (the rows a chip are few: 1024 TILE rows of 64 columns at most)."""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, SP_AXIS, TP_AXIS

NCORES = 4  # cores a chip (data movement; a [NCORES, 1] row from (0, 0))
MAX_PIECES = 4
WORDS = 32  # the constant table's words a chip: [pieces, (lane, offset, count, dst) x MAX_PIECES, ..]
BATCH_BYTES = 16384  # L1 bytes of the pages read a NoC barrier

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page
constexpr uint32_t PPU = get_compile_time_arg_val(1);   // pages a unit (a row: 1; a tile row: Wt)
constexpr uint32_t B = get_compile_time_arg_val(2);     // pages a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(3);    // cores
constexpr uint32_t SLOT = (PAGE + 63) & ~63u;
constexpr auto s_args = TensorAccessorArgs<4>();
constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
constexpr auto d_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t cst = get_arg_val<uint32_t>(1);
    const uint32_t dst = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);  // this core's index of NC
    uint32_t start[4];                            // the lanes' first units (per-run values)
    for (uint32_t i = 0; i < 4; ++i) {
        start[i] = get_arg_val<uint32_t>(4 + i);
    }
    const auto S = TensorAccessor(s_args, src, PAGE);
    const auto C = TensorAccessor(c_args, cst, 128);
    const auto D = TensorAccessor(d_args, dst, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t tab = (get_write_ptr(0) + 63) & ~63u;  // this chip's pieces (128 B)
    const uint32_t buf = tab + 128;
    noc_async_read(C.get_noc_addr(0), tab, 128);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    const uint32_t np = w[0];
    uint32_t total = 0;
    for (uint32_t p = 0; p < np; ++p) {
        total += w[3 + 4 * p] * PPU;
    }
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(total) * k) / NC);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(total) * (k + 1)) / NC);
    // destination page g (in [lo, hi)) -> its piece: dst pages [dst_p PPU, (dst_p + cnt_p) PPU)
    uint32_t g = lo;
    while (g < hi) {
        uint32_t nb = 0;
        uint32_t d0 = g;
        for (; nb < B && g < hi; ++nb, ++g) {
            uint32_t sp = 0;
            for (uint32_t p = 0; p < np; ++p) {
                const uint32_t lane = w[1 + 4 * p], off = w[2 + 4 * p], cnt = w[3 + 4 * p], dd = w[4 + 4 * p];
                if (g >= dd * PPU && g < (dd + cnt) * PPU) {
                    const uint32_t j = g - dd * PPU;
                    sp = (start[lane] + off) * PPU + j;
                    break;
                }
            }
            noc_async_read(S.get_noc_addr(sp), buf + nb * SLOT, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < nb; ++t) {
            noc_async_write(buf + t * SLOT, D.get_noc_addr(d0 + t), PAGE);
        }
        noc_async_write_barrier();
    }
}
"""


def _pieces(sp: int, n: int, lanes: int, unit: int) -> list:
    """Per SP row r: [(lane, offset in the lane, count, destination offset)] in units of ``unit`` rows: the lanes'
    rows of the natural block [r R, (r + 1) R), R = n / sp."""
    R, lr = n // sp, n // lanes
    out = []
    for r in range(sp):
        ps = []
        for L in range(lanes):
            a, b = max(r * R, L * lr), min((r + 1) * R, (L + 1) * lr)
            if a < b:
                assert (a - L * lr) % unit == 0 and (b - a) % unit == 0 and (a - r * R) % unit == 0, (a, b, unit)
                ps.append((L, (a - L * lr) // unit, (b - a) // unit, (a - r * R) // unit))
        assert 1 <= len(ps) <= MAX_PIECES, ps
        out.append(ps)
    return out


class LaneRows:
    """One table's per-lane rows program (``lane_rows``)."""

    def __init__(self, mesh, table: ttnn.Tensor, n: int, lanes: int):
        self.mesh = mesh
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        self.n, self.lanes = int(n), int(lanes)
        self.R = self.n // self.sp
        assert self.R * self.sp == self.n and self.lanes <= 4, (n, self.sp, lanes)
        self.width = int(table.shape[-1])
        self.dtype, self.layout = table.dtype, table.layout
        tile = table.layout == ttnn.TILE_LAYOUT
        self.unit = 32 if tile else 1
        self.ppu = self.width // 32 if tile else 1
        self.page = (
            ttnn.tile_size(table.dtype) if tile else self.width * {ttnn.bfloat16: 2, ttnn.float32: 4}[table.dtype]
        )
        assert self.R % self.unit == 0, (self.R, self.unit)
        words = torch.zeros(self.sp, 1, 1, WORDS, dtype=torch.int32)
        for r, ps in enumerate(_pieces(self.sp, self.n, self.lanes, self.unit)):
            words[r, 0, 0, 0] = len(ps)
            for p, (L, off, cnt, dd) in enumerate(ps):
                words[r, 0, 0, 1 + 4 * p : 5 + 4 * p] = torch.tensor([L, off, cnt, dd], dtype=torch.int32)
        from . import ccl
        from .attention import _upload  # (load-time-like constant of the geometry, made once)

        self.const = _upload(mesh, words, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, None))
        ccl.pin(mesh, self.const)
        self.batch = max(1, BATCH_BYTES // (-(-self.page // 64) * 64))
        self._pp = None

    def _out(self) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.R, self.width]), self.dtype, self.layout, self.mesh, DRAM
        )

    def _build(self, table, out):
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NCORES - 1, 0))])
        ct = [self.page, self.ppu, self.batch, NCORES]
        for t in (table, self.const, out):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 64 + 128 + self.batch * (-(-self.page // 64) * 64)
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
            )
        ]
        rt = ttnn.RuntimeArgs()
        addrs = [int(table.buffer_address()), int(self.const.buffer_address()), int(out.buffer_address())]
        for k in range(NCORES):
            rt[k][0] = addrs + [k, 0, 0, 0, 0]
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
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(self.sp - 1, self.tp - 1))] = prog
        return md

    def __call__(self, table: ttnn.Tensor, starts) -> ttnn.Tensor:
        """Each chip its SP row's rows of the lanes' rows [s, s + n / lanes) of ``table`` -> [1, 1, n / SP, width] in the
        table's dtype / layout (a new tensor)."""
        assert tuple(table.shape)[-1] == self.width and table.dtype == self.dtype and table.layout == self.layout
        lr = self.n // self.lanes
        vals = [int(s) for s in starts] + [0] * (4 - len(starts))
        for s in vals[: self.lanes]:
            assert s % self.unit == 0 and 0 <= s and s + lr <= int(table.shape[-2]), (s, lr, table.shape)
        out = self._out()
        io = [table, self.const, out]
        if self._pp is None:
            self._pp = ttnn.bringup.prepare_program(self._build(table, out), io, [(0, 4 + i) for i in range(4)])
            want = NCORES * (3 + 4)  # (table, const, out and the 4 starts a core)
            assert self._pp.patch_count == want, f"LaneRows: {self._pp.patch_count} patched slots, {want} expected"
        ttnn.bringup.run_prepared(self._pp, io, [v // self.unit for v in vals])
        return out


_BY: dict = {}  # (id(mesh), table shape, dtype, layout, n, lanes) -> (mesh, LaneRows)


def lane_rows(table: ttnn.Tensor, starts, n: int) -> ttnn.Tensor:
    """PACKED_HOST_FAST=on: ``mesh_partition(lane_slices(table, starts, n / len(starts)), SP)`` in one program (the
    same rows on every chip; ``starts`` a tt/rope.LaneStarts, n the rows of every lane together)."""
    mesh = table.device()
    key = (id(mesh), tuple(table.shape), table.dtype, table.layout, int(n), len(starts))
    hit = _BY.get(key)
    if hit is None or hit[0] is not mesh:
        hit = _BY[key] = (mesh, LaneRows(mesh, table, n, len(starts)))
    return hit[1](table, starts)
