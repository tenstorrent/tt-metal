# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Functional test for the binary_ng Metal 2.0 / DataflowBuffer (DFB) port of the ResNet50
residual-add config.

ResNet50-WH's only binary op is the residual `ttnn.add_(out, ds_out, activations=[RELU])` (see the
residual add in models/.../ttnn_functional_resnet50.py): ADD, no broadcast, bf8 (also bf16),
HEIGHT/BLOCK sharded L1, in-place, fused RELU. No other test combines all of these, so this covers
the gated Metal 2.0 DFB path producing correct results.

Run with the DFB path ENABLED:
    TT_METAL_BINARY_NG_METAL_V2=1 TT_METAL_SLOW_DISPATCH_MODE=1 \
        pytest tests/ttnn/unit_tests/operations/eltwise/test_binary_ng_resnet_add.py

With the flag unset the same ops run via the descriptor path (regression coverage).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _height_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _block_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# PCC thresholds: bf16 is tight; bf8_b loses mantissa bits so it is looser (the descriptor path uses
# the same relaxed bound for bf8 sharded add). We NEVER weaken these below what the descriptor path
# achieves for the identical config.
_PCC = {ttnn.bfloat16: 0.9997, ttnn.bfloat8_b: 0.9920}


def _run_resnet_add(device, dtype_tt, mem_config_fn, shard_shape, core_grid, total_shape, fuse_relu):
    torch.manual_seed(0)

    a_pt = torch.randn(total_shape, dtype=torch.bfloat16)
    b_pt = torch.randn(total_shape, dtype=torch.bfloat16)

    mem_config = mem_config_fn(shard_shape, core_grid)

    a_tt = ttnn.from_torch(a_pt, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)] if fuse_relu else []

    # In-place add_ : out aliases a_tt (the ResNet residual pattern).
    out_tt = ttnn.add_(a_tt, b_tt, activations=activations)

    golden = torch.add(a_pt, b_pt)
    if fuse_relu:
        golden = torch.relu(golden)

    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC[dtype_tt])
    return out_tt


# A single height-sharded core column (1x8 grid): 8 shards of one 32x32 tile each.
_HEIGHT_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 7))})
_HEIGHT_SHARD = [32, 32]
_HEIGHT_SHAPE = torch.Size([8 * 32, 32])

# A block-sharded 2x4 grid: shard [32, 32] over an 8-shard block layout.
_BLOCK_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))})
_BLOCK_SHARD = [32, 32]
_BLOCK_SHAPE = torch.Size([2 * 32, 4 * 32])


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("fuse_relu", [False, True])
def test_resnet_add_height_sharded(device, dtype_tt, fuse_relu):
    _run_resnet_add(
        device,
        dtype_tt,
        _height_sharded_config,
        _HEIGHT_SHARD,
        _HEIGHT_GRID,
        _HEIGHT_SHAPE,
        fuse_relu,
    )


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("fuse_relu", [False, True])
def test_resnet_add_block_sharded(device, dtype_tt, fuse_relu):
    _run_resnet_add(
        device,
        dtype_tt,
        _block_sharded_config,
        _BLOCK_SHARD,
        _BLOCK_GRID,
        _BLOCK_SHAPE,
        fuse_relu,
    )


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_resnet_add_multitile_shard(device, dtype_tt):
    # Larger shards (4 tiles per core) to exercise the compute chunk loop, fused RELU on.
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))})  # 4 cores
    shard = [4 * 32, 32]  # 4 tiles tall per shard
    shape = torch.Size([4 * 4 * 32, 32])
    _run_resnet_add(device, dtype_tt, _height_sharded_config, shard, grid, shape, fuse_relu=True)


def test_resnet_add_program_cache_hit(device):
    """
    A 2nd dispatch must hit the program cache and still be correct (the borrowed sharded DFB
    addresses are refreshed on cache hit via the adapter's SetProgramRunArgs path). Run the same
    in-place add_ twice on freshly-allocated inputs and verify both dispatches.
    """
    torch.manual_seed(0)
    mem_config = _height_sharded_config(_HEIGHT_SHARD, _HEIGHT_GRID)
    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]

    num_before = device.num_program_cache_entries()

    for i in range(2):
        a_pt = torch.randn(_HEIGHT_SHAPE, dtype=torch.bfloat16) + float(i)  # distinct inputs per run
        b_pt = torch.randn(_HEIGHT_SHAPE, dtype=torch.bfloat16)
        a_tt = ttnn.from_torch(
            a_pt, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )
        b_tt = ttnn.from_torch(
            b_pt, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )
        out_tt = ttnn.add_(a_tt, b_tt, activations=activations)
        golden = torch.relu(torch.add(a_pt, b_pt))
        assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC[ttnn.bfloat8_b])

    # The 2nd dispatch must NOT add a new cache entry (program-cache hit).
    num_after = device.num_program_cache_entries()
    assert num_after - num_before == 1, f"expected 1 new cache entry across 2 dispatches, got {num_after - num_before}"
