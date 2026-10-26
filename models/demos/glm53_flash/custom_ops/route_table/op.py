# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RouteTable``: the regions MoE's routing tables in two data-movement programs (tt-blaze G53F-0195).

tt/moe.py ``_forward_regions`` builds them with ~27 small ttnn ops per sub-chunk (one-hot of the dense routing matrix,
a triangular-matmul rank, tile-aligned counts and their cumsum, the slot formula, two gathers at the picks, the uint32
row copies of counts and offsets). This op computes the same values from the router's top-k ids and weights alone:

* pass 1 (``kernels/dm_route_counts.cpp``): per 32-token block, the picks of each local expert;
* pass 2 (``kernels/dm_route_slots.cpp``): every core sums the block rows (prefix + totals) and walks its blocks in
  token order, writing each pick's region slot and its masked weight; core 0 writes counts / offsets.

Outputs (bit-identical to the composite for the router's weights, which are positive normal fp32 at the picks):
``pos`` [1,1,T,k] fp32 TILE (slot, dummy = ``rows - 1`` off-chip), ``w_loc`` [1,1,T,k] fp32 TILE (w at the local picks,
+0 elsewhere), ``counts`` / ``offsets`` [1, E] uint32 ROW_MAJOR. A pick is local when its weight is a nonzero normal
fp32 (the composite's ``dense != 0`` with dense = bf16(w) at the picks) and its id is in the chip's contiguous expert
range (``table >= 0``, the dispatch table). Contract: ``idx`` / ``w`` [1,1,T,k] fp32 TILE, T % 32 == 0, k <= 16;
``table`` [1, E] int32 ROW_MAJOR.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _even_split(n: int, grid) -> tuple[list, list[tuple[int, int]]]:
    ncores = min(n, grid.x * grid.y)
    base, extra = divmod(n, ncores)
    cores, spans, b0 = [], [], 0
    for i in range(ncores):
        k = base + (1 if i < extra else 0)
        cores.append(ttnn.CoreCoord(i % grid.x, i // grid.x))
        spans.append((b0, k))
        b0 += k
    return cores, spans


class RouteTable:
    CB_SCR = 0

    def __init__(self, device, e_local: int):
        self.device = device
        self.e_local = e_local
        self._parts: dict = {}

    def _scratch(self, crs, size):
        return ttnn.CBDescriptor(
            total_size=size,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=self.CB_SCR, data_format=ttnn.uint32, page_size=size)
            ],
        )

    def __call__(self, idx: ttnn.Tensor, w: ttnn.Tensor, table: ttnn.Tensor, rows: int):
        T, k = int(idx.shape[-2]), int(idx.shape[-1])
        E = int(table.shape[-1])
        assert idx.dtype == ttnn.float32 and idx.layout == ttnn.TILE_LAYOUT, (idx.dtype, idx.layout)
        assert w.dtype == ttnn.float32 and w.layout == ttnn.TILE_LAYOUT and tuple(w.shape) == tuple(idx.shape)
        assert table.dtype == ttnn.int32 and table.layout == ttnn.ROW_MAJOR_LAYOUT, (table.dtype, table.layout)
        assert T % TILE == 0 and k <= 16, (T, k)
        dev, E_loc = self.device, self.e_local
        nblocks = T // TILE
        tile_bytes = idx.buffer_aligned_page_size()
        table_bytes = table.buffer_aligned_page_size()
        assert tile_bytes == TILE * TILE * 4 and table_bytes % 64 == 0, (tile_bytes, table_bytes)
        # per-block local counts, each row padded to a 64 B multiple (Blackhole DRAM alignment)
        e_pad = -(-(E_loc * 4) // 64) * 64 // 4
        dram = ttnn.DRAM_MEMORY_CONFIG
        bc = ttnn.allocate_tensor_on_device(ttnn.Shape([nblocks, e_pad]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, dram)
        row_bytes = bc.buffer_aligned_page_size()
        assert row_bytes == e_pad * 4, (row_bytes, e_pad)
        pos = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, k]), ttnn.float32, ttnn.TILE_LAYOUT, dev, dram)
        wl = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, k]), ttnn.float32, ttnn.TILE_LAYOUT, dev, dram)
        counts = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, dram)
        offsets = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, dram)
        assert counts.buffer_aligned_page_size() == table_bytes

        # G53F-0206: both programs' per-geometry parts are built once; a call sets the buffer addresses (common
        # runtime args) and a precomputed program hash (GLM53_CUSTOM_OP_CACHE=0 rebuilds them every call, the A/B)
        key = (T, k, E, rows)
        cached = os.environ.get("GLM53_CUSTOM_OP_CACHE", "1") == "1"
        parts = self._parts.get(key) if cached else None
        if parts is None:
            parts = self._build(idx, w, table, bc, pos, wl, counts, offsets, rows)
            self._parts[key] = parts
        (k1, cbs1, h1), (k2, cbs2, h2) = parts
        k1.common_runtime_args = [idx.buffer_address(), w.buffer_address(), table.buffer_address(), bc.buffer_address()]
        p1 = ttnn.ProgramDescriptor(kernels=[k1], semaphores=[], cbs=cbs1)
        p1.custom_program_hash = h1
        ttnn.generic_op([idx, w, table, bc], p1)
        k2.common_runtime_args = [
            idx.buffer_address(),
            w.buffer_address(),
            table.buffer_address(),
            bc.buffer_address(),
            pos.buffer_address(),
            wl.buffer_address(),
            counts.buffer_address(),
            offsets.buffer_address(),
        ]
        p2 = ttnn.ProgramDescriptor(kernels=[k2], semaphores=[], cbs=cbs2)
        p2.custom_program_hash = h2
        ttnn.generic_op([idx, w, table, bc, pos, wl, counts, offsets], p2)
        ttnn.deallocate(bc)
        return pos, wl, counts, offsets

    def _build(self, idx, w, table, bc, pos, wl, counts, offsets, rows):
        T, k = int(idx.shape[-2]), int(idx.shape[-1])
        E = int(table.shape[-1])
        dev, E_loc = self.device, self.e_local
        nblocks = T // TILE
        tile_bytes = idx.buffer_aligned_page_size()
        table_bytes = table.buffer_aligned_page_size()
        row_bytes = bc.buffer_aligned_page_size()
        grid = dev.compute_with_storage_grid_size()
        cores, spans = _even_split(nblocks, grid)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        # pass 1: per-block local counts
        ct1 = [k, E, E_loc, self.CB_SCR, tile_bytes, table_bytes, row_bytes]
        for t in (idx, w, table, bc):
            ct1 += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        ct1 += [nblocks, len(cores), grid.x]  # the kernels derive their block spans from their position (G53F-0206)
        rt1 = ttnn.RuntimeArgs()  # no per-core runtime args
        k1 = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dm_route_counts.cpp"),
            core_ranges=crs,
            compile_time_args=ct1,
            runtime_args=rt1,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs1 = [self._scratch(crs, 2 * tile_bytes + table_bytes + row_bytes)]

        # pass 2: slots, masked weights, counts / offsets
        ct2 = [k, E, E_loc, nblocks, rows - 1, self.CB_SCR, tile_bytes, table_bytes, row_bytes]
        for t in (idx, w, table, bc, pos, wl, counts, offsets):
            ct2 += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        ct2 += [nblocks, len(cores), grid.x]
        rt2 = ttnn.RuntimeArgs()
        k2 = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dm_route_slots.cpp"),
            core_ranges=crs,
            compile_time_args=ct2,
            runtime_args=rt2,
            config=ttnn.ReaderConfigDescriptor(),
        )
        scratch2 = 4 * tile_bytes + 3 * table_bytes + 2 * row_bytes + nblocks * row_bytes
        cbs2 = [self._scratch(crs, scratch2)]
        spans_t = tuple((c.x, c.y, b0, n) for c, (b0, n) in zip(cores, spans))
        h1 = hash(("glm53_route_counts", tuple(ct1), spans_t)) & ((1 << 63) - 1)
        h2 = hash(("glm53_route_slots", tuple(ct2), spans_t, scratch2)) & ((1 << 63) - 1)
        return (k1, cbs1, h1), (k2, cbs2, h2)
