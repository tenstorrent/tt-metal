# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``UntilizeUsed``: ``ttnn.untilize`` of the regions MoE's expert output over its USED rows only (tt-blaze G53F-0203).

The regions buffer ``[R, H]`` (``R = max_buf``, bf8 TILE) holds the chip's local regions packed from row 0 in expert
order: the rows below ``run = offsets[E-1] + align32(counts[E-1])`` (the expert kernel's own counts / offsets) carry
the expert outputs; the rows past it are never read by the fused combine (``custom_ops/gather_combine_reduce``) except
the dummy row ``R - 1`` (a token's pick 0 when it is not local, weight 0). ``ttnn.untilize`` converts all R rows
(``R = T*k + E_loc*32``: ~3.4x the local picks at the 8x4 geometry); this op reads ``run`` on the device and splits the
``run / 32`` used tile rows evenly over all its cores, so its time follows the local picks:

* reader (``kernels/reader_untilize_used.cpp``): counts / offsets -> its tile rows; the count to the compute through a
  one-page CB; each tile row's ``Wt`` tiles read in flight;
* compute (``kernels/compute_untilize_used.cpp``): ttnn.untilize's kernel_lib untilize over full-width blocks;
* writer (``kernels/writer_untilize_used.cpp``): 32 row pages per block; core 0 writes the dummy row as zeros.

Output: ``[R, H]`` bf16 ROW_MAJOR, equal to ``ttnn.untilize(out)`` on rows ``[0, run)`` (bf8 -> bf16 is exact either
way: same unpack, same pack), zeros on row ``R - 1``, unwritten elsewhere. Contract: ``out`` bf8 TILE DRAM interleaved
``[R, H]`` (R, H multiples of 32), ``counts`` / ``offsets`` ``[1, E]`` uint32 ROW_MAJOR (the expert kernel's inputs).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class UntilizeUsed:
    CB_IN, CB_CNT, CB_SCR_R, CB_SCR_W, CB_OUT = 0, 1, 2, 3, 16

    def __init__(self, device):
        self.device = device

    @staticmethod
    def _cb(crs, index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    def __call__(self, out: ttnn.Tensor, counts: ttnn.Tensor, offsets: ttnn.Tensor) -> ttnn.Tensor:
        y, program = self.descriptor(out, counts, offsets)
        return ttnn.generic_op([out, counts, offsets, y], program)

    def descriptor(self, out: ttnn.Tensor, counts: ttnn.Tensor, offsets: ttnn.Tensor, y: ttnn.Tensor | None = None):
        """(output, ProgramDescriptor) of one call (``y`` given: reuse that output; the generic-op overhead bench,
        tests/test_generic_op_overhead.py)."""
        R, H = int(out.shape[-2]), int(out.shape[-1])
        E = int(counts.shape[-1])
        assert out.dtype == ttnn.bfloat8_b and out.layout == ttnn.TILE_LAYOUT, (out.dtype, out.layout)
        assert R % TILE == 0 and H % TILE == 0, (R, H)
        for t in (counts, offsets):
            assert t.dtype == ttnn.uint32 and t.layout == ttnn.ROW_MAJOR_LAYOUT and int(t.shape[-1]) == E
        dev = self.device
        Wt = H // TILE
        tile_bytes = out.buffer_aligned_page_size()
        table_bytes = counts.buffer_aligned_page_size()
        assert offsets.buffer_aligned_page_size() == table_bytes
        if y is None:
            y = ttnn.allocate_tensor_on_device(
                ttnn.Shape([R, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
            )
        row_bytes = y.buffer_aligned_page_size()
        assert row_bytes == H * 2, (row_bytes, H)

        grid = dev.compute_with_storage_grid_size()
        ncores = grid.x * grid.y
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

        reader_ct = [E, Wt, ncores, self.CB_IN, self.CB_CNT, self.CB_SCR_R, tile_bytes, table_bytes]
        for t in (out, counts, offsets):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [E, Wt, ncores, self.CB_OUT, self.CB_SCR_W, row_bytes, table_bytes, R - 1]
        for t in (y, counts, offsets):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [Wt, self.CB_IN, self.CB_OUT, self.CB_CNT]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            reader_rt[c.x][c.y] = [out.buffer_address(), counts.buffer_address(), offsets.buffer_address(), i]
            writer_rt[c.x][c.y] = [y.buffer_address(), counts.buffer_address(), offsets.buffer_address(), i]
            compute_rt[c.x][c.y] = []

        out_tile_bytes = TILE * TILE * 2
        cbs = [
            self._cb(crs, self.CB_IN, ttnn.bfloat8_b, tile_bytes, 2 * Wt),
            self._cb(crs, self.CB_OUT, ttnn.bfloat16, out_tile_bytes, 2 * Wt),
            self._cb(crs, self.CB_CNT, ttnn.uint32, 64, 1),
            self._cb(crs, self.CB_SCR_R, ttnn.uint32, 2 * table_bytes, 1),
            self._cb(crs, self.CB_SCR_W, ttnn.uint32, 2 * table_bytes + row_bytes, 1),
        ]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_untilize_used.cpp"),
                core_ranges=crs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_untilize_used.cpp"),
                core_ranges=crs,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_untilize_used.cpp"),
                core_ranges=crs,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False),
            ),
        ]
        return y, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
