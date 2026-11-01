# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RegionPlan``: the regions prefill path's bookkeeping (``TtRoutedExperts._forward_regions``) as one single-core
``ttnn.generic_op``. From the router's top-k expert ids (uint32 ROW_MAJOR [1, T*k], row-major (t, j)) and the chip's
local expert list (``gidx`` uint32 ROW_MAJOR [1, E_loc], global ids) it writes

* ``counts`` [1, E] uint32: the chip's picks per global expert (0 off-chip);
* ``offsets`` [1, E] uint32: the tile-aligned exclusive prefix of the counts (every local expert's region starts at
  a 32-row boundary);
* ``pos`` [1, T*k] uint32: the region slot of every pick (``offsets[e]`` + the pick's rank among e's picks in
  (t, j) order), ``max_buf - 1`` (the dummy slot) at the non-local picks;
* ``loc`` [1, 1, T, k] fp32 TILE: 1.0 at the local picks (the routing-weight mask of ``post_combine_reduce``).

It replaces the one-hot / ``cumsum`` / gather chain (bookkeeping 27 ms + pick tables 8 ms per MoE layer at 16 K
gathered tokens on the 8x4, 2026-11-01). The ids are read twice in 64 KB chunks (counts, then ranks). Golden:
``RegionPlan.golden``. Contract in README.md.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNEL = Path(__file__).resolve().parent / "kernels" / "region_plan.cpp"
TILE = 32
CHUNK_ENTRIES = 16384  # 64 KB of ids per pass chunk: whole tile rows for any k dividing 16384 / 32
DUMMY = 0xFFFFFFFF


class RegionPlan:
    def __init__(self, device, *, core: ttnn.CoreCoord | None = None):
        self.device = device
        self.core = core if core is not None else ttnn.CoreCoord(0, 0)

    @staticmethod
    def golden(ids: torch.Tensor, gidx: torch.Tensor, E: int, max_buf: int):
        """ids [T, k] (int), gidx [E_loc] (the chip's global expert ids) -> counts [E], offsets [E], pos [T, k],
        loc [T, k] (float 0/1), the exact arithmetic of ``_forward_regions``."""
        T, k = ids.shape
        local = torch.zeros(E, dtype=torch.bool)
        local[gidx.long()] = True
        flat = ids.reshape(-1).long()
        is_loc = local[flat.clamp(0, E - 1)] & (flat < E)
        counts = torch.zeros(E, dtype=torch.int64)
        counts.index_add_(0, flat[is_loc], torch.ones(int(is_loc.sum()), dtype=torch.int64))
        aligned = ((counts + TILE - 1) // TILE) * TILE
        offsets = torch.cumsum(aligned, 0) - aligned
        pos = torch.full((T * k,), max_buf - 1, dtype=torch.int64)
        rank = torch.zeros(E, dtype=torch.int64)
        for i in torch.nonzero(is_loc).reshape(-1).tolist():
            e = int(flat[i])
            pos[i] = int(offsets[e]) + int(rank[e])
            rank[e] += 1
        return counts, offsets, pos.reshape(T, k), is_loc.float().reshape(T, k)

    def allocate_outputs(self, T: int, k: int, E: int):
        dev, dram = self.device, ttnn.DRAM_MEMORY_CONFIG
        u32, rm = ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        counts = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), u32, rm, dev, dram)
        offsets = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), u32, rm, dev, dram)
        pos = ttnn.allocate_tensor_on_device(ttnn.Shape([1, T * k]), u32, rm, dev, dram)
        loc = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, k]), ttnn.float32, ttnn.TILE_LAYOUT, dev, dram)
        return counts, offsets, pos, loc

    def __call__(
        self, ids: ttnn.Tensor, gidx: ttnn.Tensor, counts, offsets, pos, loc, *, max_buf: int, E: int, T: int, k: int
    ):
        """ids [1, T*k] uint32 ROW_MAJOR (one page); gidx [1, E_loc] uint32 ROW_MAJOR; the four pre-allocated
        outputs of ``allocate_outputs``. ``T`` a multiple of 32, ``k`` <= 16, ``T * k`` a multiple of
        ``CHUNK_ENTRIES`` or smaller than it (whole tile rows per chunk). Returns ``(counts, offsets, pos, loc)``."""
        assert ids.layout == ttnn.ROW_MAJOR_LAYOUT and ids.dtype == ttnn.uint32, (ids.layout, ids.dtype)
        assert gidx.layout == ttnn.ROW_MAJOR_LAYOUT and gidx.dtype == ttnn.uint32
        assert list(ids.shape)[-1] == T * k and T % TILE == 0 and 1 <= k <= 16, (list(ids.shape), T, k)
        assert CHUNK_ENTRIES % (TILE * k) == 0, (CHUNK_ENTRIES, k)
        E_loc = int(gidx.shape[-1])
        assert list(loc.shape) == [1, 1, T, k] and loc.layout == ttnn.TILE_LAYOUT and loc.dtype == ttnn.float32
        assert list(counts.shape) == [1, E] and list(offsets.shape) == [1, E] and list(pos.shape) == [1, T * k]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])

        def cb(i, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)],
            )

        e_pad = -(-E * 4 // 64) * 64 + 64
        cbs = [
            cb(0, 64, CHUNK_ENTRIES * 4 // 64 + 2),  # the ids chunk
            cb(1, 64, CHUNK_ENTRIES * 4 // 64 + 2),  # the pos chunk
            cb(2, 64, 5 * e_pad // 64 + (-(-E_loc * 4 // 64)) + 4),  # map, offsets, ranks, counts, gidx
            cb(3, 4096, 2),  # one fp32 tile (+ alignment slack)
        ]
        ct = [E, E_loc, T, k, CHUNK_ENTRIES]
        for t in (ids, gidx, counts, offsets, pos, loc):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [
            ids.buffer_address(),
            gidx.buffer_address(),
            counts.buffer_address(),
            offsets.buffer_address(),
            pos.buffer_address(),
            loc.buffer_address(),
            max_buf,
            ids.buffer_aligned_page_size(),
            gidx.buffer_aligned_page_size(),
            counts.buffer_aligned_page_size(),
        ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL),
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
        ttnn.generic_op([ids, gidx, counts, offsets, pos, loc], program)
        return counts, offsets, pos, loc
