# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RowGather``: ``out[r] = src[index[r]]`` over ROW_MAJOR tensors whose rows are one DRAM page each (bf16 [S, W]
and [N, W]), as a model-local ``ttnn.generic_op`` (one data-movement kernel on every core: a contiguous range of
output rows per core, a ring of in-flight 8 KB row reads then the writes). It replaces the two ``ttnn.embedding``
row gathers of the regions prefill path (``TtRoutedExperts._forward_regions``: the token rows into the region
buffer, the kernel's output rows at the picks), which ran at 43-57 GB/s (27 + 18 ms per MoE layer at 16 K gathered
tokens, 2026-11-01 07:45). Indices at or beyond ``S`` write a zero row. Contract in README.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.demos.glm53_flash.custom_ops.core_ranges import rect_core_ranges

KERNEL = Path(__file__).resolve().parent / "kernels" / "reader_row_gather.cpp"
ROWS_IN_FLIGHT = int(os.environ.get("GLM53_ROW_GATHER_IN_FLIGHT", "8"))
ALIGN_ENTRIES = 16  # per-core row ranges start at multiples of 16 index entries (a 64 B NoC read offset)


class RowGather:
    def __init__(self, device, *, rows_in_flight: int = ROWS_IN_FLIGHT):
        self.device = device
        self.rows_in_flight = rows_in_flight
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = grid.x, grid.y

    @staticmethod
    def golden(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """src [S, W], index [N] (any integer dtype) -> [N, W]; out-of-range indices give zero rows."""
        idx = index.reshape(-1).long()
        ok = (idx >= 0) & (idx < src.shape[0])
        out = src[idx.clamp(0, src.shape[0] - 1)].clone()
        out[~ok] = 0
        return out

    def __call__(self, src: ttnn.Tensor, index: ttnn.Tensor, out: ttnn.Tensor) -> ttnn.Tensor:
        """src [S, W] bf16 ROW_MAJOR (one page per row); index [1, N] uint32 ROW_MAJOR (one page); out [N, W] bf16
        ROW_MAJOR, pre-allocated. Returns ``out``."""
        assert src.layout == ttnn.ROW_MAJOR_LAYOUT and out.layout == ttnn.ROW_MAJOR_LAYOUT, (src.layout, out.layout)
        assert src.dtype == out.dtype == ttnn.bfloat16, (src.dtype, out.dtype)
        assert index.layout == ttnn.ROW_MAJOR_LAYOUT and index.dtype == ttnn.uint32, (index.layout, index.dtype)
        S, W = int(src.shape[-2]), int(src.shape[-1])
        N = int(index.shape[-1])
        assert list(index.shape)[:-1] == [1] * (len(index.shape) - 1), list(index.shape)
        assert list(out.shape)[-2:] == [N, W], (list(out.shape), N, W)
        row_bytes = W * 2
        assert src.buffer_aligned_page_size() == row_bytes == out.buffer_aligned_page_size(), (
            src.buffer_aligned_page_size(),
            out.buffer_aligned_page_size(),
            row_bytes,
        )
        idx_page_bytes = index.buffer_aligned_page_size()
        assert idx_page_bytes >= N * 4, (idx_page_bytes, N)
        # the output rows in contiguous ranges, every range start a multiple of ALIGN_ENTRIES
        n_cores = max(1, min(self.grid_x * self.grid_y, -(-N // ALIGN_ENTRIES)))
        per = -(-N // n_cores)
        per = -(-per // ALIGN_ENTRIES) * ALIGN_ENTRIES
        ranges = []
        r = 0
        while r < N:
            ranges.append((r, min(per, N - r)))
            r += per
        cores = [ttnn.CoreCoord(c // self.grid_y, c % self.grid_y) for c in range(len(ranges))]
        core_set = rect_core_ranges(cores)
        k = self.rows_in_flight
        idx_slice_bytes = -(-(per * 4) // 64) * 64 + 64

        def cb(i, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)],
            )

        cbs = [
            cb(0, row_bytes, k + 1),  # the ring of row buffers (+1 row of 64 B alignment slack)
            cb(1, 64, idx_slice_bytes // 64 + 1),  # the core's index slice
            cb(2, row_bytes, 2),  # a zero row (+ slack)
        ]
        ct = [row_bytes, k, S]
        for t in (src, index, out):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for core, (r0, n) in zip(cores, ranges):
            rt[core.x][core.y] = [
                r0,
                n,
                src.buffer_address(),
                index.buffer_address(),
                out.buffer_address(),
                idx_page_bytes,
            ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL),
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
        return ttnn.generic_op([src, index, out], program)
