# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RowSumSq``: ``ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)`` in one pass, bit-identical (tt-blaze G53F-0209).

The prefill mHC projection (``deepseek_v3_d_p/tt/mhc/tt_mhc.py`` ``_project``) squares the fp32 residual streams
``x`` ([T, 4 x 4096] per chip) into a fresh tensor and sums its rows: at 2048 rows that is a 134 MB write and two
134 MB reads per wrap (0.9 + 0.43 ms, G53F-0206's split), 90 wraps per 16 K chunk. This op reads ``x`` once and
squares each tile in dest before folding it, with the composite's own LLK calls in its order (see
``kernels/compute_row_sumsq.cpp``), so the result is the composite's bit for bit.

* reader (``kernels/reader_row_sumsq.cpp``): the core's tile rows, each row's Wt tiles in order;
* compute (``kernels/compute_row_sumsq.cpp``): copy, ``mul_binary_tile`` square, ``add_binary_tile`` fold, then the
  intra-tile ``sfpu_reduce`` row sum;
* writer (``kernels/writer_row_sumsq.cpp``): one tile per tile row.

Contract: ``x`` fp32 TILE DRAM interleaved ``[..., T, W]`` (T, W multiples of 32, leading dims 1). Output
``[..., T, 1]`` fp32 TILE (column 0 of each tile = the row sums, as ``ttnn.sum(..., keepdim=True)``). A whole tile
row runs on one core (the composite's order), so up to T / 32 cores work.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
GROUP = 8  # tiles per reader barrier
NUM_CBS = 64  # unpack_to_dest_mode entries


class RowSumSq:
    CB_IN, CB_OUT = 0, 16

    def __init__(self, device, cached: bool = True):
        self.device = device
        self.cached = cached
        self._parts: dict = {}

    @staticmethod
    def _cb(crs, index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        y, program = self.descriptor(x)
        return ttnn.generic_op([x, y], program)

    def descriptor(self, x: ttnn.Tensor, y: ttnn.Tensor | None = None):
        shape = list(x.shape)
        T, W = int(shape[-2]), int(shape[-1])
        assert x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
        assert T % TILE == 0 and W % TILE == 0 and all(int(d) == 1 for d in shape[:-2]), shape
        if y is None:
            y = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape[:-1] + [1]), ttnn.float32, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
            )
        key = (T, W)
        cached = self.cached and os.environ.get("GLM53_CUSTOM_OP_CACHE", "1") == "1"  # read per call (A/B)
        parts = self._parts.get(key) if cached else None
        if parts is None:
            parts = self._build(x, y)
            self._parts[key] = parts
        reader, writer, compute, cbs, program_hash = parts
        reader.common_runtime_args = [x.buffer_address()]
        writer.common_runtime_args = [y.buffer_address()]
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        program.custom_program_hash = program_hash
        return y, program

    def _build(self, x, y):
        T, W = int(x.shape[-2]), int(x.shape[-1])
        Ht, Wt = T // TILE, W // TILE
        assert Wt % GROUP == 0, (Wt, GROUP)
        tile_bytes = x.buffer_aligned_page_size()
        assert tile_bytes == TILE * TILE * 4 and y.buffer_aligned_page_size() == tile_bytes
        grid = self.device.compute_with_storage_grid_size()
        ncores = grid.x * grid.y
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

        reader_ct = [Ht, Wt, ncores, grid.x, GROUP, self.CB_IN, tile_bytes]
        reader_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        writer_ct = [Ht, ncores, grid.x, self.CB_OUT, tile_bytes]
        writer_ct += ttnn.TensorAccessorArgs(y).get_compile_time_args()
        compute_ct = [Ht, Wt, ncores, grid.x, self.CB_IN, self.CB_OUT]
        cbs = [
            self._cb(crs, self.CB_IN, ttnn.float32, tile_bytes, 2 * GROUP),
            self._cb(crs, self.CB_OUT, ttnn.float32, tile_bytes, 2),
        ]
        compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        modes[self.CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader_row_sumsq.cpp"),
            core_ranges=crs,
            compile_time_args=reader_ct,
            runtime_args=ttnn.RuntimeArgs(),
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer_row_sumsq.cpp"),
            core_ranges=crs,
            compile_time_args=writer_ct,
            runtime_args=ttnn.RuntimeArgs(),
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "compute_row_sumsq.cpp"),
            core_ranges=crs,
            compile_time_args=compute_ct,
            runtime_args=ttnn.RuntimeArgs(),
            config=compute_cfg,
        )
        program_hash = hash(("glm53_row_sumsq", T, W, tuple(reader_ct), tuple(writer_ct), tuple(compute_ct)))
        return reader, writer, compute, cbs, program_hash & ((1 << 63) - 1)
