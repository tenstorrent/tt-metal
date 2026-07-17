# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Functional test for the Metal 2.0 / DataflowBuffer (DFB) path of `ttnn.experimental.quasar.add_`,
the ResNet50 residual-add config.

ResNet50's residual add is `add_(out, ds_out, activations=[RELU])` (see the residual add in
models/.../resnet50/quasar/tt/ttnn_functional_resnet50.py): ADD, no broadcast, HEIGHT/BLOCK sharded
L1, in-place, fused RELU. The op routes this config through the DFB factory automatically (no env
flag) — that path is arch-portable, so the same test runs on real Wormhole (CB-backed DFB) and on
the Quasar simulator (overlay-backed DFB).

The shapes use bf16 and core grids that fit both Wormhole (8x8) and the Quasar simulator (8x4). The
WH model config uses bf8, but Quasar's DFB data-format validation accepts bf16/MX formats rather
than the Bfp8_b block-float, so the Quasar-portable test uses bf16.

Run on Wormhole:
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_resnet_add.py

Run on the Quasar simulator:
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 ARCH_NAME=quasar CHIP_ARCH=quasar \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_resnet_add.py
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


# PCC thresholds. We NEVER weaken these below what the descriptor path achieves for the same config.
_PCC = {ttnn.bfloat16: 0.9997}


def _run_resnet_add(device, dtype_tt, mem_config_fn, shard_shape, core_grid, total_shape, fuse_relu):
    torch.manual_seed(0)

    a_pt = torch.randn(total_shape, dtype=torch.bfloat16)
    b_pt = torch.randn(total_shape, dtype=torch.bfloat16)

    mem_config = mem_config_fn(shard_shape, core_grid)

    a_tt = ttnn.from_torch(a_pt, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)] if fuse_relu else []

    # In-place add_ : out aliases a_tt (the ResNet residual pattern).
    out_tt = ttnn.experimental.quasar.add_(a_tt, b_tt, activations=activations)

    golden = torch.add(a_pt, b_pt)
    if fuse_relu:
        golden = torch.relu(golden)

    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC[dtype_tt])
    return out_tt


# A single height-sharded core column: 4 shards of one 32x32 tile each. A 4-tall column (y=0..3)
# fits both Wormhole (8x8) and the Quasar simulator (8x4) grids.
_HEIGHT_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))})
_HEIGHT_SHARD = [32, 32]
_HEIGHT_SHAPE = torch.Size([4 * 32, 32])

# A block-sharded 4x2 grid (x=0..3, y=0..1): shard [32, 32] over an 8-shard block layout. Fits 8x4.
_BLOCK_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))})
_BLOCK_SHARD = [32, 32]
_BLOCK_SHAPE = torch.Size([2 * 32, 4 * 32])

# Uneven height shard: the tensor is 3 tiles tall but the shard is 2 tiles tall over a 2-core column,
# so the last core holds 1 logical tile + 1 allocated padding tile. The selector admits this (matching
# a/b/c specs) and the DFB factory processes the full rounded-up shard (2 tiles) on every core. The
# padding tile is allocated L1 with no host page mapped, so over-processing it is in-bounds and the
# logical output stays correct. A 2-core column fits both Wormhole (8x8) and the Quasar simulator (8x4).
_UNEVEN_HEIGHT_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))})  # 2 cores
_UNEVEN_HEIGHT_SHARD = [2 * 32, 32]  # 2 tiles tall per shard
_UNEVEN_HEIGHT_SHAPE = torch.Size([3 * 32, 32])  # 3 tiles total -> end core: 1 logical + 1 padding

# Uneven block shard: a 2x2 shard ([64, 64]) over a 2x2 core grid, but the tensor is 3x3 tiles, so the
# boundary-row and boundary-column cores carry partial shards (the corner core is partial in both dims).
# Same partial-end-core mechanism as the height case, exercised on the 2D block layout. Fits 8x4.
_UNEVEN_BLOCK_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 1))})  # 2x2 cores
_UNEVEN_BLOCK_SHARD = [2 * 32, 2 * 32]  # 2x2 tiles per shard
_UNEVEN_BLOCK_SHAPE = torch.Size([3 * 32, 3 * 32])  # 3x3 tiles -> boundary cores partial


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16])
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


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16])
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


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16])
def test_resnet_add_multitile_shard(device, dtype_tt):
    # Larger shards (4 tiles per core) to exercise the compute chunk loop, fused RELU on. The 4-core
    # column (y=0..3) fits both Wormhole (8x8) and the Quasar simulator (8x4).
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))})  # 4 cores
    shard = [4 * 32, 32]  # 4 tiles tall per shard
    shape = torch.Size([4 * 4 * 32, 32])
    _run_resnet_add(device, dtype_tt, _height_sharded_config, shard, grid, shape, fuse_relu=True)


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16])
@pytest.mark.parametrize("fuse_relu", [False, True])
def test_resnet_add_uneven_height_sharded(device, dtype_tt, fuse_relu):
    # Uneven height shard (partial end core). The selector admits it because a/b/c share one shard spec;
    # the DFB factory over-processes the full rounded-up shard into allocated padding, so the logical
    # output must still match the golden add.
    _run_resnet_add(
        device,
        dtype_tt,
        _height_sharded_config,
        _UNEVEN_HEIGHT_SHARD,
        _UNEVEN_HEIGHT_GRID,
        _UNEVEN_HEIGHT_SHAPE,
        fuse_relu,
    )


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16])
@pytest.mark.parametrize("fuse_relu", [False, True])
def test_resnet_add_uneven_block_sharded(device, dtype_tt, fuse_relu):
    # Uneven block shard: boundary-row/column cores carry partial shards (corner core partial in both
    # dims). Same partial-end-core handling on the 2D block layout.
    _run_resnet_add(
        device,
        dtype_tt,
        _block_sharded_config,
        _UNEVEN_BLOCK_SHARD,
        _UNEVEN_BLOCK_GRID,
        _UNEVEN_BLOCK_SHAPE,
        fuse_relu,
    )


def test_resnet_add_program_cache_hit(device):
    """
    A 2nd dispatch must hit the program cache and still be correct (the borrowed sharded DFB
    addresses are refreshed on cache hit via the adapter's SetProgramRunArgs path). Run the same
    in-place add twice on freshly-allocated inputs and verify both dispatches.
    """
    torch.manual_seed(0)
    mem_config = _height_sharded_config(_HEIGHT_SHARD, _HEIGHT_GRID)
    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]

    num_before = device.num_program_cache_entries()

    for i in range(2):
        a_pt = torch.randn(_HEIGHT_SHAPE, dtype=torch.bfloat16) + float(i)  # distinct inputs per run
        b_pt = torch.randn(_HEIGHT_SHAPE, dtype=torch.bfloat16)
        a_tt = ttnn.from_torch(
            a_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )
        b_tt = ttnn.from_torch(
            b_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )
        out_tt = ttnn.experimental.quasar.add_(a_tt, b_tt, activations=activations)
        golden = torch.relu(torch.add(a_pt, b_pt))
        assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC[ttnn.bfloat16])

    # The 2nd dispatch must NOT add a new cache entry (program-cache hit).
    num_after = device.num_program_cache_entries()
    assert num_after - num_before == 1, f"expected 1 new cache entry across 2 dispatches, got {num_after - num_before}"
