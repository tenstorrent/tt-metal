# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2d debug tests. DO NOT DELETE.

Lever #1 (wide-W CB chunking of the general cross-core path): verify identity is
preserved for wide-W crossovers AND that the per-core CB footprint is bounded by
a constant (2*Wt_chunk*tile), NOT by Wt. Exercises both npr==1 (HEIGHT-sharded,
full-width input page) and npr>1 (WIDTH-sharded, width-split pages spanning a
subset of shard pages per chunk).
"""

import pytest
import torch
import ttnn

from ttnn.operations.tilize.tilize_program_descriptor import (
    _create_general_program_descriptor,
    _pick_wt_chunk,
    WT_CHUNK_MAX,
    TILE_W,
)

L1_BUDGET = 1499136


def _grid_crs(n):
    """A 1xN core range set (n cores in a row)."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


def _height_mc(n_cores, folded_h, w):
    shard = [folded_h // n_cores, w]  # HEIGHT: full width, split along height
    sp = ttnn.ShardSpec(_grid_crs(n_cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sp)


def _width_mc(n_cores, folded_h, w):
    shard = [folded_h, w // n_cores]  # WIDTH: full height, split along width
    sp = ttnn.ShardSpec(_grid_crs(n_cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sp)


def _cb_bound_check(input_tensor, output_tensor, w):
    """Build the general descriptor and assert both CBs are bounded by
    2*Wt_chunk*tile — constant in W, NOT the old 2*Wt*tile."""
    wt = w // TILE_W
    wt_chunk = _pick_wt_chunk(wt)
    pd = _create_general_program_descriptor(input_tensor, output_tensor, use_multicore=True)
    sizes = [cb.total_size for cb in pd.cbs]
    in_tile = ttnn.tile_size(input_tensor.dtype)
    out_tile = ttnn.tile_size(output_tensor.dtype)
    expected = {2 * wt_chunk * in_tile, 2 * wt_chunk * out_tile}
    for s in sizes:
        assert s in expected, f"CB size {s} not in expected constant-bounded set {expected} (wt_chunk={wt_chunk})"
        # The whole point: constant in W. The un-chunked size would be Wt/Wt_chunk x bigger.
        assert s < 2 * wt * max(in_tile, out_tile), f"CB {s} still scales with Wt={wt}"
    assert wt_chunk <= WT_CHUNK_MAX
    return wt_chunk, sizes


# --- Done-when gate: wide-W HEIGHT crossover (npr==1), 8 cores, W=2048 (Wt=64) ---
def test_2d_wide_w_height_crossover_cb_bounded(device):
    N, C, H, W = 1, 1, 512, 2048
    x = torch.rand([N, C, H, W], dtype=torch.bfloat16)
    in_mc = _height_mc(8, N * C * H, W)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)

    # allocate the interleaved TILE output tensor to build the descriptor for the CB check
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([N, C, H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    wt_chunk, sizes = _cb_bound_check(t, out, W)
    assert wt_chunk == 8, wt_chunk  # 64 % 8 == 0

    res = ttnn.to_torch(ttnn.tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_multicore=True))
    assert torch.equal(x, res), f"HEIGHT wide-W max_diff={(x.float()-res.float()).abs().max()}"


# --- npr>1 overlap loop: WIDTH-sharded wide input, chunk spans 2 shard pages ---
def test_2d_width_sharded_npr_gt1_crossover(device):
    # W=1024, 8 cores -> shard_w=128 (npr=8). Wt=32, wt_chunk=8 -> chunk_width=8 tiles
    # = 512B (bf16); shard_page=128*2=256B, so each chunk spans 2 shard pages.
    N, C, H, W = 1, 1, 256, 1024
    x = torch.rand([N, C, H, W], dtype=torch.bfloat16)
    in_mc = _width_mc(8, N * C * H, W)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([N, C, H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    _cb_bound_check(t, out, W)

    res = ttnn.to_torch(ttnn.tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_multicore=True))
    assert torch.equal(x, res), f"WIDTH npr>1 max_diff={(x.float()-res.float()).abs().max()}"


# --- fp32 wide-W HEIGHT crossover (fp32 lossless path + chunking) ---
def test_2d_wide_w_height_crossover_fp32(device):
    N, C, H, W = 1, 1, 256, 2048
    x = torch.rand([N, C, H, W], dtype=torch.float32)
    in_mc = _height_mc(8, N * C * H, W)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    res = ttnn.to_torch(ttnn.tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_multicore=True))
    assert torch.equal(x, res), f"fp32 wide-W max_diff={(x.float()-res.float()).abs().max()}"


# --- L1-bound sanity: an EXTREME width that would OOM un-chunked stays bounded ---
def test_2d_extreme_width_cb_under_budget(device):
    # W=8192 -> Wt=256. Un-chunked CB (bf16) = 2*256*2048 = 1.0MB each -> 2 CBs
    # = 2MB > 1.5MB L1 -> would OOM. Chunked: 2*8*2048 = 32KB each. Just build the
    # descriptor and confirm the footprint is a small constant (no device launch).
    N, H, W = 1, 256, 8192
    x = torch.rand([N, H, W], dtype=torch.bfloat16)
    in_mc = _height_mc(8, N * H, W)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([N, H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    wt_chunk, sizes = _cb_bound_check(t, out, W)
    assert sum(sizes) < L1_BUDGET, f"total CB {sum(sizes)} exceeds L1 budget"
    assert sum(sizes) < 100 * 1024, f"expected small constant footprint, got {sum(sizes)}"
