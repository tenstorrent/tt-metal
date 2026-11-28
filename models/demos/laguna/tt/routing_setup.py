# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Token-dispatch routing metadata for Laguna-S as two generic_ops (see kernels/routing_*.cpp).

Laguna's experts are sharded one dispatch group per chip (dispatch_group_size 1), so each chip's own histogram is
the whole group's: no all_gather. The two steps match ttnn.experimental.deepseek_prefill's masked_bincount and
offset_cumsum with that group size:

  1. masked_bincount: per-expert token counts of this chip's top-k indices, counting only experts the dispatch
     table maps to this chip (entry >= 0). 64 cores (8x8) each count a contiguous range of token rows (BRISC and
     NCRISC each half of it, into their own L1 histogram), then the first ceil(E / 16) cores each sum one
     16-expert column of every core's histogram (flat reduction).
  2. offset_cumsum over the one-row [1, E] histogram, on one core: for expert e,
       totals[e]        = histogram[e]
       expert_region[e] = sum over the earlier experts j of e's chip group of ceil(totals[j] / 32) * 32
       offsets[e]       = expert_region[e] (the one source device's local offset is 0)
     and all_offsets is offsets' one row.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_GRID_X = _GRID_Y = 8  # masked_bincount's fixed 64-core grid
_EXPERTS_PER_REDUCER = 16  # experts one reducer core sums (one 64-byte histogram column)


class RoutingOutputs(NamedTuple):
    global_dispatch_offsets: ttnn.Tensor
    total_counts_per_expert: ttnn.Tensor
    expert_region_offsets: ttnn.Tensor
    expert_histograms: ttnn.Tensor
    all_global_dispatch_offsets: ttnn.Tensor


def _cb(index, fmt, page, total, cores):
    return ttnn.CBDescriptor(
        total_size=total,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def _accessors(*tensors):
    args = []
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    return args


def masked_bincount(indices, expert_mask, n_routed_experts, num_experts_per_token):
    """indices: [T, >= K] uint16 TILE interleaved (the router's top-k); expert_mask: [1, E(+1)] int32 row-major
    (>= 0: expert on this chip). Returns [E] uint32 row-major DRAM: this chip's tokens per expert."""
    device = indices.device()
    assert indices.dtype == ttnn.uint16 and indices.layout == ttnn.TILE_LAYOUT, (indices.dtype, indices.layout)
    tokens = indices.padded_shape[0]
    num_cores = _GRID_X * _GRID_Y
    assert tokens % num_cores == 0, f"token count {tokens} must divide over the {num_cores}-core grid"
    num_reducers = -(-n_routed_experts // _EXPERTS_PER_REDUCER)
    assert num_reducers <= num_cores, n_routed_experts
    tile_h = 32
    rows = tokens // num_cores
    h_brisc = rows // 2
    h_ncrisc = rows - h_brisc
    # a RISC's row range can straddle one extra tile
    max_tiles = [(h + tile_h - 1) // tile_h + 1 for h in (h_brisc, h_ncrisc)]

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([n_routed_experts]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    in_page = indices.buffer_aligned_page_size()
    out_page = out.buffer_aligned_page_size()
    mask_page = expert_mask.buffer_aligned_page_size()

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_GRID_X - 1, _GRID_Y - 1))})
    cores = [ttnn.CoreCoord(x, y) for y in range(_GRID_Y) for x in range(_GRID_X)]  # row-major, as the op orders them
    noc_xy = [device.worker_core_from_logical_core(c) for c in cores]
    collector = noc_xy[0]
    column_bytes = _EXPERTS_PER_REDUCER * 4
    cbs = [
        _cb(0, ttnn.uint16, in_page, max_tiles[0] * in_page, grid),  # BRISC input tiles
        _cb(1, ttnn.uint32, out_page, out_page, grid),  # local histogram
        _cb(2, ttnn.uint16, in_page, max_tiles[1] * in_page, grid),  # NCRISC input tiles
        _cb(3, ttnn.uint32, out_page, out_page, grid),  # NCRISC's histogram, then a reducer's 16 counts
        _cb(4, ttnn.int32, mask_page, mask_page, grid),  # expert dispatch table
        _cb(5, ttnn.uint32, column_bytes, num_cores * column_bytes, grid),  # reducer staging
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=0) for i in range(3)]  # init, done, gather
    acc = _accessors(indices, out, expert_mask)

    def ct(cb_in, h, is_initializer):
        return [
            cb_in,
            1,
            in_page,
            out_page,
            h,
            num_experts_per_token,
            n_routed_experts,
            is_initializer,
            0,
            1,
            2,
            3,
            collector.x,
            collector.y,
            num_cores,
            4,
            mask_page,
            tile_h,
        ] + acc

    addrs = [indices.buffer_address(), out.buffer_address(), expert_mask.buffer_address()]
    all_xy = [v for p in noc_xy for v in (p.x, p.y)]
    rt_brisc, rt_ncrisc = [], []
    for i, c in enumerate(cores):
        first_row = i * rows
        reducer = i if i < num_reducers else 0xFFFFFFFF
        rt_brisc.append(((c.x, c.y), addrs + [first_row, num_cores, num_reducers, reducer] + all_xy))
        rt_ncrisc.append(((c.x, c.y), addrs + [first_row + h_brisc, 0]))

    def kernel(ct_args, rt, processor, noc):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "routing_masked_bincount.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct_args,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
        )

    kernels = [
        kernel(ct(0, h_brisc, 1), rt_brisc, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        kernel(ct(2, h_ncrisc, 0), rt_ncrisc, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
    ]
    ttnn.generic_op([indices, expert_mask, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    return out


def offset_cumsum(histogram, experts_per_chip):
    """histogram: [E] uint32 row-major, this chip's (= its one-chip dispatch group's) counts.
    Returns (offsets, totals, expert_region, all_offsets), each [1, E] uint32 row-major DRAM."""
    device = histogram.device()
    W = histogram.shape[-1]
    H = 1
    assert W % experts_per_chip == 0, (W, experts_per_chip)
    src = ttnn.reshape(histogram, ttnn.Shape([H, W]))
    outs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, W]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(4)
    ]
    offsets, totals, region, all_offsets = outs
    in_page = src.buffer_aligned_page_size()
    pages = [t.buffer_aligned_page_size() for t in outs]
    assert pages[3] == pages[0], pages  # the kernel stages offsets in all_offsets' row buffer
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    cbs = [
        _cb(0, ttnn.uint32, in_page, H * in_page, core),  # all H input rows
        _cb(1, ttnn.uint32, pages[0], pages[0], core),
        _cb(2, ttnn.uint32, pages[3], H * pages[3], core),  # every all_offsets row
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "routing_offset_cumsum.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=[0, 1, 2, in_page] + pages + [W, H, experts_per_chip] + _accessors(src, *outs),
        # row_idx 0: the chip is the only device of its dispatch group
        runtime_args=[((0, 0), [src.buffer_address()] + [t.buffer_address() for t in outs] + [0])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([src, offsets, totals, region, all_offsets], ttnn.ProgramDescriptor(kernels=[reader], cbs=cbs))
    return offsets, totals, region, all_offsets


class LagunaRoutingSetup:
    """Per-chip routing metadata for token dispatch with one-chip dispatch groups (Laguna-S on a 1 x D mesh)."""

    def __init__(self, mesh_device, expert_dispatch_table: torch.Tensor, experts_per_chip: int):
        """expert_dispatch_table: [num_dispatch_groups, E(+1)] int32, entry >= 0 where the group hosts the expert;
        row g goes to mesh column g."""
        self.experts_per_chip = experts_per_chip
        self.expert_mask = ttnn.from_torch(
            expert_dispatch_table,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 0)),
        )

    def __call__(self, ttnn_top_k_experts_indices, num_routed_experts, num_experts_per_tok) -> RoutingOutputs:
        indices = ttnn_top_k_experts_indices
        if indices.layout != ttnn.TILE_LAYOUT:
            indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
        if len(indices.shape) == 3:
            indices = ttnn.squeeze(indices, 0)
        histogram = masked_bincount(indices, self.expert_mask, num_routed_experts, num_experts_per_tok)
        offsets, totals, region, all_offsets = offset_cumsum(histogram, self.experts_per_chip)
        return RoutingOutputs(offsets, totals, region, histogram, all_offsets)
