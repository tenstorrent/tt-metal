# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RegionScatter``: the regions MoE's region buffer written row by row from the pick slots (tt-blaze G53F-0193).

    buf[pos[t, j], :] = x[t, :]      for every pick with pos[t, j] != R - 1 (the dummy slot of the non-local picks)

The regions path (tt/moe.py ``_forward_regions``) builds the same buffer in three steps -- ``pos`` to a uint32 row copy
(typecast + untilize of the tiny [T, k] slot table), ``ttnn.scatter`` of a slot -> token map, ``ttnn.embedding`` of
every one of the R rows (~3/4 of them dummy / padding, copies of token 0) -- 31.7 ms per MoE layer per 16 K chunk in
G53F-0192's op table. This op reads the slots from pos's own fp32 TILE layout and writes only the local picks' rows.
The rows it leaves unwritten (region padding up to a tile, the rows past the regions, the dummy slot) are never read
as data that reaches the output: the expert kernel computes each row of a region on its own (row-independent matmuls,
per-row bfp8 blocks), and the combine reads only local picks' slots and pick 0, whose dummy-slot row the expert kernel
never writes in either path. So every row the output depends on is bit-identical.

Contract: ``x`` [T, H] bf16 ROW_MAJOR DRAM interleaved (one page per row); ``pos`` [1, 1, T, k] fp32 TILE (integer slot
values < 2^24, distinct at the local picks), T % 32 == 0, k <= 16; returns ``buf`` [R, H] bf16 ROW_MAJOR (``R`` =
``rows``), its unwritten rows uninitialised.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
NB = 8  # tokens staged per batch (8 x 8 KB rows of L1)


class RegionScatter:
    CB_ROWS, CB_POS = 0, 1

    def __init__(self, device):
        self.device = device

    def __call__(self, x: ttnn.Tensor, pos: ttnn.Tensor, rows: int) -> ttnn.Tensor:
        T, H = int(x.shape[-2]), int(x.shape[-1])
        k = int(pos.shape[-1])
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.ROW_MAJOR_LAYOUT, (x.dtype, x.layout)
        assert pos.dtype == ttnn.float32 and pos.layout == ttnn.TILE_LAYOUT, (pos.dtype, pos.layout)
        assert int(pos.shape[-2]) == T and T % TILE == 0 and k <= 16, (pos.shape, T, k)
        row_bytes = x.buffer_aligned_page_size()
        assert row_bytes == H * 2, (row_bytes, H)
        buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([rows, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        assert buf.buffer_aligned_page_size() == row_bytes
        pos_page = pos.buffer_aligned_page_size()

        grid = self.device.compute_with_storage_grid_size()
        blocks = T // TILE
        ncores = min(blocks, grid.x * grid.y)
        base, extra = divmod(blocks, ncores)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cbs = [
            ttnn.CBDescriptor(
                total_size=NB * row_bytes,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=self.CB_ROWS, data_format=ttnn.bfloat16, page_size=row_bytes)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=pos_page,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=self.CB_POS, data_format=ttnn.float32, page_size=pos_page)
                ],
            ),
        ]
        ct = [k, row_bytes, rows - 1, self.CB_ROWS, self.CB_POS, NB, pos_page]
        for t in (x, pos, buf):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        b0 = 0
        for i, c in enumerate(cores):
            n = base + (1 if i < extra else 0)
            rt[c.x][c.y] = [x.buffer_address(), pos.buffer_address(), buf.buffer_address(), b0, n]
            b0 += n
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "dm_region_scatter.cpp"),
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, pos, buf], program)
