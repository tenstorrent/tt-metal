# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_height, input_width, input_memory_layout, input_sharded_memory_config_args, output_sharded_memory_config_args, input_override, output_override",
    [
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            dict(core_grid=ttnn.CoreGrid(y=4, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=4, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=4), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=4), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            2304,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            None,
            None,
        ),
        (
            32,
            16,
            ttnn.ROW_MAJOR_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            7168,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=7, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            [32, 128],
            None,
        ),
    ],
)
def test_reshard(
    device,
    input_height,
    input_width,
    input_memory_layout,
    input_sharded_memory_config_args,
    output_sharded_memory_config_args,
    input_override,
    output_override,
):
    input_shape = [1, 1, input_height, input_width]

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    interleaved_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=input_memory_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_override == None:
        input_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)
    else:
        input_shard_memory_config = ttnn.create_sharded_memory_config(
            input_override, **input_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
        )

    if output_override == None:
        output_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **output_sharded_memory_config_args)
    else:
        output_shard_memory_config = ttnn.create_sharded_memory_config(
            output_override, **output_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
        )

    # interleaved_to_sharded
    sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)

    # reshard
    sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)

    # sharded_to_interleaved
    interleaved_output_tensor = ttnn.to_memory_config(sharded_output_tensor, ttnn.DRAM_MEMORY_CONFIG)

    output = ttnn.to_torch(interleaved_output_tensor)

    assert_with_pcc(torch_input_tensor, output, 1.0)


@pytest.mark.parametrize(
    "input_shape, input_shard_shape,  input_sharded_memory_config_args",
    [
        (
            [1, 1, 32, 1024],
            [32, 256],
            dict(
                core_grid=ttnn.experimental.tensor.CoreRangeSet(
                    {
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, 0), ttnn.experimental.tensor.CoreCoord(0, 3)
                        ),
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        ),
        (
            [1, 1, 32, 1024],
            [32, 128],
            dict(
                core_grid=ttnn.experimental.tensor.CoreRangeSet(
                    {
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, 0), ttnn.experimental.tensor.CoreCoord(0, 1)
                        ),
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, 2), ttnn.experimental.tensor.CoreCoord(0, 3)
                        ),
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, 4), ttnn.experimental.tensor.CoreCoord(0, 7)
                        ),
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        ),
    ],
)
def test_shard_with_corerangeset(device, input_shape, input_shard_shape, input_sharded_memory_config_args):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    interleaved_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_shard_memory_config = ttnn.create_sharded_memory_config(
        input_shard_shape, **input_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
    )
    # interleaved_to_sharded
    sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)

    # sharded_to_interleaved
    interleaved_output_tensor = ttnn.to_memory_config(sharded_input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    output = ttnn.to_torch(interleaved_output_tensor)

    assert_with_pcc(torch_input_tensor, output, 1.0)


@pytest.mark.parametrize(
    "shape, strategy, orientation, core_grid",
    [
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.COLUMN_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.COLUMN_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 128, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=4)),
        ([1, 1, 1024, 128], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR, ttnn.CoreGrid(y=4, x=2)),
    ],
)
def test_create_sharded_memory_config(device, shape, strategy, orientation, core_grid):
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        input_data,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    shard_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=core_grid,
        strategy=strategy,
        orientation=orientation,
        use_height_and_width_as_shard_shape=False,
    )

    x_t = ttnn.to_memory_config(x, memory_config=shard_config, dtype=ttnn.bfloat16)
    output_data = ttnn.from_device(x_t)
    output_data = ttnn.to_torch(output_data)

    passing = torch.equal(input_data, output_data)
    assert passing
