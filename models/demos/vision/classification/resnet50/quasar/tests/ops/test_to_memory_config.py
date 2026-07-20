# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for ttnn.experimental.quasar.to_memory_config as used by resnet50/quasar.

to_memory_config re-lays a tensor across a different memory layout (interleaved <-> sharded, or a
resharding) WITHOUT changing shape or values, so a to_torch of the output must round-trip the input
exactly (PCC ~1.0). This isolates the resharding call-sites so the LLK team can test/fix the data
movement alone.

Call-sites mirrored (see ttnn_functional_resnet50.py). There are 7 to_memory_config calls; several
are gated on is_wormhole_b0()/is_blackhole() and never run on Quasar. The two that DO run on Quasar
are the width-shard transitions feeding avg_pool / fc (run(), ~lines 996 and 1027):
    x = ttnn.experimental.quasar.to_memory_config(x, width_mem_config)   # interleaved/sharded -> WIDTH_SHARDED
and the residual reshard in the bottleneck (resnet50Bottleneck.__call__, ~line 395):
    ds_out = ttnn.experimental.quasar.to_memory_config(ds_out, out.memory_config())  # sharded -> sharded

Representative transitions covered here:
  - interleaved DRAM (TILE) -> HEIGHT_SHARDED L1  (the reshard pattern into a sharded conv/pool)
  - interleaved DRAM (TILE) -> WIDTH_SHARDED  L1  (the avg_pool / fc feed)
  - HEIGHT_SHARDED L1       -> interleaved DRAM    (the reverse / gather)

The core grid is derived from the device so this runs on a full Quasar part or a tiny emulator grid.

Run (craq-sim example):
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
    pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_to_memory_config.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _fit_num_cores(num_tiles, grid):
    """Largest core count <= device grid that evenly divides num_tiles (keeps shards tile-aligned)."""
    cap = grid.x * grid.y
    n = min(cap, num_tiles)
    while n > 1 and num_tiles % n != 0:
        n -= 1
    return n


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_to_memory_config_interleaved_to_height_sharded(mesh_device):
    torch.manual_seed(0)
    device = mesh_device

    # (1, 1, H, W) with H a multiple of 32 (tile) so it height-shards evenly across cores.
    grid = device.compute_with_storage_grid_size()
    height_tiles = 32  # 32 tiles = 1024 rows
    width = 256
    num_cores = _fit_num_cores(height_tiles, grid)
    shard_height = (height_tiles // num_cores) * 32
    input_shape = (1, 1, height_tiles * 32, width)

    x = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out = ttnn.experimental.quasar.to_memory_config(tt_in, sharded_mem_config)

    # Landed in the requested sharded layout, same shape.
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert tuple(out.shape) == tuple(input_shape)

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    # Reshard preserves values -> exact round-trip.
    assert_with_pcc(x, got, pcc=0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_to_memory_config_interleaved_to_width_sharded(mesh_device):
    torch.manual_seed(0)
    device = mesh_device

    # WIDTH_SHARDED feed for avg_pool / fc: width split across cores, one tile row high.
    grid = device.compute_with_storage_grid_size()
    width_tiles = 32  # 32 tiles = 1024 cols (mirrors the 1024/2048-wide fc-region tensors)
    height = 32
    num_cores = _fit_num_cores(width_tiles, grid)
    shard_width = (width_tiles // num_cores) * 32
    input_shape = (1, 1, height, width_tiles * 32)

    x = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    width_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, height, shard_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out = ttnn.experimental.quasar.to_memory_config(tt_in, width_mem_config)

    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert tuple(out.shape) == tuple(input_shape)

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    assert_with_pcc(x, got, pcc=0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_to_memory_config_sharded_to_interleaved(mesh_device):
    torch.manual_seed(0)
    device = mesh_device

    # The reverse gather: HEIGHT_SHARDED L1 -> interleaved DRAM (value-preserving).
    grid = device.compute_with_storage_grid_size()
    height_tiles = 32
    width = 256
    num_cores = _fit_num_cores(height_tiles, grid)
    shard_height = (height_tiles // num_cores) * 32
    input_shape = (1, 1, height_tiles * 32, width)

    x = torch.rand(input_shape, dtype=torch.bfloat16)
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    out = ttnn.experimental.quasar.to_memory_config(tt_in, ttnn.DRAM_MEMORY_CONFIG)

    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert tuple(out.shape) == tuple(input_shape)

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    assert_with_pcc(x, got, pcc=0.999)
