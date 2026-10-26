# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RowGather``: ``out[0, i] = table[idx[0, i]]`` for a row-major bf16 table, as a model-local ``ttnn.generic_op``
(tt-blaze G53F-0186). Bitwise the same result as ``ttnn.embedding(idx, table, layout=ROW_MAJOR)``, which the prefill
MoE regions path calls twice per sub-chunk; that op's row-major reader waits on every row read (one 8 KB read in
flight per core, ~60 GB/s measured), this one keeps ``batch`` reads in flight per core and overlaps the writes of one
batch with the reads of the next (a reader and a writer kernel on a two-batch CB).

Contract: ``table`` [R, H] (or [1, R, H]) bf16 ROW_MAJOR DRAM interleaved (one page per row); ``idx`` [1, N] uint32
ROW_MAJOR (one page), every id < R; returns [1, N, H] bf16 ROW_MAJOR DRAM interleaved."""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.glm53_flash.custom_ops.core_ranges import rect_core_ranges

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
UNIT = 16  # rows per work unit: a core's first id sits on a 64 B boundary of the index page


class RowGather:
    CB_ROWS, CB_IDX = 0, 1

    def __init__(self, device, *, batch: int = 8, idx_chunk: int = 1024):
        assert idx_chunk % batch == 0 and idx_chunk % UNIT == 0, (batch, idx_chunk)
        self.device = device
        self.batch, self.idx_chunk = batch, idx_chunk
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = grid.x, grid.y

    @staticmethod
    def golden(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        t = table.reshape(-1, table.shape[-1])
        return t[idx.reshape(-1).long()].reshape(1, -1, t.shape[-1])

    def __call__(self, idx: ttnn.Tensor, table: ttnn.Tensor) -> ttnn.Tensor:
        assert table.layout == ttnn.ROW_MAJOR_LAYOUT and table.dtype == ttnn.bfloat16, (table.layout, table.dtype)
        assert idx.layout == ttnn.ROW_MAJOR_LAYOUT and idx.dtype == ttnn.uint32, (idx.layout, idx.dtype)
        H = int(table.shape[-1])
        N = int(idx.shape[-1])
        assert int(idx.shape[0]) == 1 and len(idx.shape) == 2, idx.shape
        row_bytes = table.buffer_aligned_page_size()
        assert row_bytes == H * 2, (row_bytes, H)
        idx_page = idx.buffer_aligned_page_size()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, N, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        assert out.buffer_aligned_page_size() == row_bytes
        units = -(-N // UNIT)
        n_cores = min(units, self.grid_x * self.grid_y)
        cores = [ttnn.CoreCoord(c // self.grid_y, c % self.grid_y) for c in range(n_cores)]
        core_set = rect_core_ranges(cores)
        B = self.batch
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * B * row_bytes,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=self.CB_ROWS, data_format=ttnn.bfloat16, page_size=row_bytes)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=self.idx_chunk * 4 + 64,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=self.CB_IDX, data_format=ttnn.uint32, page_size=self.idx_chunk * 4 + 64
                    )
                ],
            ),
        ]
        reader_ct = [self.CB_ROWS, self.CB_IDX, row_bytes, B, self.idx_chunk, idx_page]
        reader_ct += ttnn.TensorAccessorArgs(table).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(idx).get_compile_time_args()
        writer_ct = [self.CB_ROWS, row_bytes, B] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        per, extra = divmod(units, n_cores)
        u = 0
        for c, core in enumerate(cores):
            nu = per + (1 if c < extra else 0)
            row0, row1 = u * UNIT, min((u + nu) * UNIT, N)
            u += nu
            reader_rt[core.x][core.y] = [table.buffer_address(), idx.buffer_address(), row0, row1 - row0]
            writer_rt[core.x][core.y] = [out.buffer_address(), row0, row1 - row0]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_row_gather.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_row_gather.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([table, idx, out], program)
