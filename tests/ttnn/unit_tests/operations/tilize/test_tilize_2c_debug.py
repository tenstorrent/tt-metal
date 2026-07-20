# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2c debug tests. DO NOT DELETE.

Isolates the 2c sub-levers with the exact golden shapes so each can be
debugged independently of the full golden matrix.
"""

import pytest
import torch
import ttnn


# --- Lever 2: cliff/padded same-spec nd (input spec == output spec) ---
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        ([4, 128, 128], [2, 64, 64]),  # even, no pad (2b baseline)
        ([3, 160, 160], [2, 64, 64]),  # cliff: 18 shards / 4 cores + padding
        ([5, 4, 160, 160], [2, 3, 64, 96]),  # 36 / 4 = 9, padded dims
        ([23, 96, 160], [4, 64, 96]),  # 24 / 4 = 6, cliff on dim0/dim2
    ],
)
def test_2c_samespec_cliff_nd(device, tensor_shape, shard_shape):
    torch.manual_seed(42)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    nd = ttnn.NdShardSpec(shard_shape=shard_shape, grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd)

    x = torch.rand(tensor_shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    out = ttnn.tilize(t, memory_config=mc, use_multicore=True)
    res = ttnn.to_torch(out)
    assert torch.equal(x, res), f"mismatch shape={tensor_shape} max_diff={(x.float()-res.float()).abs().max()}"


_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})  # 4 cores


def _nd(shard):
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(shard_shape=shard, grid=_GRID, orientation=ttnn.ShardOrientation.ROW_MAJOR),
    )


# --- General path: interleaved DRAM <-> nd sharded crossover (npr=1 / npr>1) ---
@pytest.mark.parametrize("use_multicore", [True, False])
def test_2c_crossover_interleaved_to_nd(device, use_multicore):
    x = torch.rand([4, 128, 128], dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    res = ttnn.to_torch(ttnn.tilize(t, memory_config=_nd([2, 64, 64]), use_multicore=use_multicore))
    assert torch.equal(x, res), f"max_diff={(x.float()-res.float()).abs().max()}"


def test_2c_crossover_nd_to_interleaved(device):
    x = torch.rand([4, 128, 128], dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_nd([2, 64, 64])
    )
    res = ttnn.to_torch(ttnn.tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_multicore=True))
    assert torch.equal(x, res), f"max_diff={(x.float()-res.float()).abs().max()}"


# --- General path: cross-spec nd -> nd (different shard spec) ---
def test_2c_crossspec_nd_to_nd(device):
    x = torch.rand([4, 128, 128], dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_nd([2, 64, 64])
    )
    res = ttnn.to_torch(ttnn.tilize(t, memory_config=_nd([1, 64, 128]), use_multicore=True))
    assert torch.equal(x, res), f"max_diff={(x.float()-res.float()).abs().max()}"


# --- General path: nd -> legacy HEIGHT/WIDTH/BLOCK ---
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_2c_nd_to_legacy(device, layout):
    x = torch.rand([4, 128, 128], dtype=torch.bfloat16)  # folded H=512, W=128, 4 cores
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_nd([2, 64, 64])
    )
    oshape = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: (128, 128),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: (512, 32),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: (256, 64),
    }[layout]
    osp = ttnn.ShardSpec(_GRID, list(oshape), ttnn.ShardOrientation.ROW_MAJOR)
    omc = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, osp)
    res = ttnn.to_torch(ttnn.tilize(t, memory_config=omc, use_multicore=True))
    assert torch.equal(x, res), f"layout={layout} max_diff={(x.float()-res.float()).abs().max()}"
