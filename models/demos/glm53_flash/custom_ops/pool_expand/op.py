# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``PoolExpand``: the DSA prefill index rows from the ranked pool ids in one data-movement pass (tt-blaze G53F-0210).

``TtIndexer._expand_pools`` turns the top-512 pool ids of every query ([1,1,T,512] uint32 ROW_MAJOR) into token ids
(4 per pool), appends the query's incomplete-pool tail ids and pads with sentinels to a multiple of 128:
typecast -> repeat_interleave -> multiply -> add -> typecast -> concat, ~6.5 ms per DSA layer-chunk at the 8x4
geometry (``score_expand`` + ``ids_concat``, G53F-0198's stage split) for integer arithmetic that is exact below 2^24.
This op computes the same integers on the data movers (``kernels/dm_pool_expand.cpp``).

Contract: ``top`` [1,1,T,K] uint32 ROW_MAJOR (pool ids < 2^22 or SENTINEL), ``tail`` [1,1,T,P-1] uint32 ROW_MAJOR, both
DRAM interleaved. Output [1,1,T,W] uint32 ROW_MAJOR, W = P*K + (P-1) rounded up to ``align`` (128).
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"


class PoolExpand:
    CB_SCR = 0

    def __init__(self, device, *, pool: int = 4, align: int = 128, cached: bool = True):
        self.device = device
        self.P = int(pool)
        self.align = int(align)
        self.cached = cached
        self._parts: dict = {}

    def width(self, K: int) -> int:
        w = self.P * K + self.P - 1
        return w + (-w) % self.align

    def __call__(self, top: ttnn.Tensor, tail: ttnn.Tensor) -> ttnn.Tensor:
        y, program = self.descriptor(top, tail)
        return ttnn.generic_op([top, tail, y], program)

    def descriptor(self, top: ttnn.Tensor, tail: ttnn.Tensor, y: ttnn.Tensor | None = None):
        T, K = int(top.shape[-2]), int(top.shape[-1])
        assert int(tail.shape[-2]) == T and int(tail.shape[-1]) == self.P - 1, (tail.shape, top.shape)
        for t in (top, tail):
            assert t.dtype == ttnn.uint32 and t.layout == ttnn.ROW_MAJOR_LAYOUT, (t.dtype, t.layout)
        W = self.width(K)
        if y is None:
            y = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, T, W]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
            )
        key = (T, K)
        cached = self.cached and os.environ.get("GLM53_CUSTOM_OP_CACHE", "1") == "1"  # read per call (A/B)
        parts = self._parts.get(key) if cached else None
        if parts is None:
            parts = self._build(top, tail, y)
            self._parts[key] = parts
        kernel, cbs, program_hash = parts
        kernel.common_runtime_args = [top.buffer_address(), tail.buffer_address(), y.buffer_address()]
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
        program.custom_program_hash = program_hash
        return y, program

    def _build(self, top, tail, y):
        T, K = int(top.shape[-2]), int(top.shape[-1])
        W = int(y.shape[-1])
        top_page, tail_page, out_page = (t.buffer_aligned_page_size() for t in (top, tail, y))
        assert top_page >= K * 4 and out_page >= W * 4, (top_page, out_page)
        grid = self.device.compute_with_storage_grid_size()
        ncores = grid.x * grid.y
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        ct = [T, ncores, grid.x, K, self.P, self.P - 1, W, self.CB_SCR, top_page, tail_page, out_page]
        for t in (top, tail, y):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        scratch = top_page + tail_page + 2 * out_page
        cbs = [
            ttnn.CBDescriptor(
                total_size=scratch,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=self.CB_SCR, data_format=ttnn.uint32, page_size=scratch)
                ],
            )
        ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dm_pool_expand.cpp"),
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=ttnn.RuntimeArgs(),
            config=ttnn.ReaderConfigDescriptor(),
        )
        program_hash = hash(("glm53_pool_expand", tuple(ct)))
        return kernel, cbs, program_hash & ((1 << 63) - 1)
