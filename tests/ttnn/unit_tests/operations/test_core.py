# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from enum import Enum


@pytest.mark.parametrize(
    "input_height, input_width, input_memory_layout, input_sharded_memory_config_args, output_sharded_memory_config_args, input_override, output_override",
    [
        (
            128,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            None,
            None,
        ),
        (
            128,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
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
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
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
            16,
            ttnn.ROW_MAJOR_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.BLOCK),
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
            1792,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=7, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            [32, 32],
            None,
        ),
        (
            32,
            7168,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=7, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            [32, 128],
            None,
        ),
        (
            32,
            320,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=1, x=5), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            8192,
            320,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=8, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=8, x=5), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        # (1, 1, 32, 8192) (32 to 8 cores width shardrd)
        (
            32,
            8192,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            (32, 256),
            None,
        ),
        # (1, 1, 32, 8192) (64 to 8 cores width shardrd)
        (
            32,
            8192,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            (32, 128),
            None,
        ),
        # (1, 1, 32, 1280) (8 to 1 cores width shardrd)
        (
            32,
            1280,
            ttnn.ROW_MAJOR_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.WIDTH),
            None,
            None,
        ),
        # (1, 1, 128, 1280) (32 cores block sharded to 4 cores height sharded)
        (
            128,
            1280,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
        (
            160,
            64,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreGrid(y=5, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dict(
                core_grid=ttnn.CoreGrid(y=2, x=2),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            (32, 64),
            (32, 96),
        ),
        (
            192,
            128,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            (96, 128),
            (128, 64),
        ),
        (
            128,
            128,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            (64, 128),
            (96, 64),
        ),
        (
            96,
            128,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2)),
                    }
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            (32, 128),
            (64, 64),
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
    if isinstance(input_sharded_memory_config_args["core_grid"], (ttnn.CoreGrid)):
        if device.core_grid.y < input_sharded_memory_config_args["core_grid"].y:
            pytest.skip()
        if device.core_grid.y < output_sharded_memory_config_args["core_grid"].y:
            pytest.skip()
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


class DirectReadWriteType(Enum):
    READ_ONLY = 0
    WRITE_ONLY = 1
    READ_WRITE = 2
    NONE = 3


@pytest.mark.parametrize(
    "data_transfer_strategy",
    [
        (DirectReadWriteType.READ_ONLY),
        (DirectReadWriteType.WRITE_ONLY),
        (DirectReadWriteType.NONE),
        (DirectReadWriteType.READ_WRITE),
    ],
)
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,  input_sharded_memory_config_args",
    [
        (
            [1, 1, 32, 1024],
            [32, 256],
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        ),
        (
            [1, 1, 32, 1024],
            [32, 128],
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 3)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 7)),
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        ),
    ],
)
def test_shard_with_corerangeset(
    device, input_shape, input_shard_shape, input_sharded_memory_config_args, data_transfer_strategy
):
    if device.core_grid.y == 7 and input_shard_shape == [32, 128]:
        pytest.skip()
    if (
        ((not (input_shape[2] % input_shard_shape[0] == 0)) or (not (input_shape[3] % input_shard_shape[1] == 0)))
        and (not (data_transfer_strategy == DirectReadWriteType.READ_WRITE))
        and (not (input_sharded_memory_config_args["strategy"] == ttnn.ShardStrategy.HEIGHT))
    ):
        pytest.skip()

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_shard_memory_config = ttnn.create_sharded_memory_config(
        input_shard_shape, **input_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
    )

    if data_transfer_strategy == DirectReadWriteType.READ_ONLY or data_transfer_strategy == DirectReadWriteType.NONE:
        interleaved_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # interleaved_to_sharded
        sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)
    else:
        sharded_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_shard_memory_config
        )

    if data_transfer_strategy == DirectReadWriteType.WRITE_ONLY or data_transfer_strategy == DirectReadWriteType.NONE:
        # sharded_to_interleaved
        interleaved_output_tensor = ttnn.to_memory_config(sharded_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.to_torch(interleaved_output_tensor)
    else:
        output = ttnn.to_torch(sharded_input_tensor)

    assert_with_pcc(torch_input_tensor, output, 1.0)


@pytest.mark.parametrize(
    "data_transfer_strategy",
    [
        (DirectReadWriteType.READ_ONLY),
        (DirectReadWriteType.WRITE_ONLY),
        (DirectReadWriteType.NONE),
        (DirectReadWriteType.READ_WRITE),
    ],
)
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,  input_sharded_memory_config_args",
    [
        (
            [1, 1, 32, 1000],
            [32, 256],
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        ),
        (
            [1, 1, 120, 1000],
            [32, 256],
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3)),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
        ),
    ],
)
def test_uneven_shard(device, input_shape, input_shard_shape, input_sharded_memory_config_args, data_transfer_strategy):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_shard_memory_config = ttnn.create_sharded_memory_config(
        input_shard_shape, **input_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
    )

    if data_transfer_strategy == DirectReadWriteType.READ_ONLY or data_transfer_strategy == DirectReadWriteType.NONE:
        interleaved_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # interleaved_to_sharded
        sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)
    else:
        sharded_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_shard_memory_config
        )
    if data_transfer_strategy == DirectReadWriteType.WRITE_ONLY or data_transfer_strategy == DirectReadWriteType.NONE:
        # sharded_to_interleaved
        interleaved_output_tensor = ttnn.to_memory_config(sharded_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.to_torch(interleaved_output_tensor)
    else:
        output = ttnn.to_torch(sharded_input_tensor)

    assert_with_pcc(torch_input_tensor, output, 1.0)


@pytest.mark.parametrize(
    "shape, strategy, orientation, core_grid",
    [
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 128, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=4)),
        ([1, 1, 1024, 128], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=4, x=2)),
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
