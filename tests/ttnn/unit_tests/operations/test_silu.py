# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from typing import Union, Tuple
from loguru import logger
import torch

# import torch.nn as nn

# import tt_lib
import tt_lib as ttl
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull
from tt_lib.utils import (
    _nearest_y,
)

TILE_WIDTH = 32


def get_shard_grid_from_num_cores(ncores: Union[int, Tuple[int, int]], device) -> ttnn.experimental.tensor.CoreRangeSet:
    max_grid_size = (8, 8) if device.arch() == ttl.device.Arch.WORMHOLE_B0 else (9, 12)  ## (y, x)
    if isinstance(ncores, int):
        if ncores % max_grid_size[1] == 0:
            core_grid = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
            grid_coord = ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, core_grid.y - 1)
            return ttnn.experimental.tensor.CoreRangeSet(
                {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
            )
        else:
            if ncores < max_grid_size[1]:
                core_grid = ttnn.CoreGrid(y=1, x=ncores)
                grid_coord = ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, 0)
                return ttnn.experimental.tensor.CoreRangeSet(
                    {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
                )
            else:
                core_grid_1 = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
                core_grid_2 = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
                grid_coord_1 = ttnn.experimental.tensor.CoreCoord(core_grid_1.x - 1, core_grid_1.y - 1)
                grid_coord_2 = ttnn.experimental.tensor.CoreCoord(core_grid_2.x - 1, core_grid_2.y - 1)
                return ttnn.experimental.tensor.CoreRangeSet(
                    {
                        ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord_1),
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, grid_coord_2.y), grid_coord_2
                        ),
                    }
                )
    elif isinstance(ncores, tuple):
        ncores_h, ncores_w = ncores
        assert ncores_h <= max_grid_size[0]
        assert ncores_w <= max_grid_size[1]
        return ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(ncores_w - 1, ncores_h - 1),
                )
            }
        )
    else:
        raise ValueError("Invalid ncores")


def run_elt_silu_relu(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    shard_grid,
    ncores,
    shard_strategy,
    shard_orientation,
    op,
):
    ## input shape is N C H W
    input_shape = [batch_size, input_channels, input_height, input_width]
    torch.manual_seed(0)
    input = torch.rand(input_shape, dtype=torch.bfloat16) * 2 - 1

    ## permute to N H W C, which is what the upsample op expects
    tt_input = input.permute(0, 2, 3, 1)
    tt_input = tt_input.reshape(1, 1, batch_size * input_height * input_width, input_channels)
    input_tensor = ttnn.from_torch(
        tt_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )
    interleaved_mem_config = ttnn.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    input_tensor = ttnn.to_memory_config(input_tensor, interleaved_mem_config)
    # input_shape = [1, 1, _nearest_y(batch_size * input_height * input_width, 32), input_channels]
    input_2d_height = input_tensor.get_legacy_shape()[2]
    input_2d_width = input_tensor.get_legacy_shape()[3]
    logger.debug(f"input_2d_height={input_2d_height} and input_2d_width={input_2d_width}")

    ## input shard
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, shard_grid[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / shard_grid[0])
        shard_width = math.ceil(input_2d_width / shard_grid[1])
        shard_orientation = ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        input_2d_height_padded = _nearest_y(input_2d_height, ncores * 32)
        shard_height = math.ceil(input_2d_height_padded / ncores)
        shard_width = input_2d_width
        shard_orientation = ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED

    assert shard_height % TILE_WIDTH == 0
    assert shard_width % TILE_WIDTH == 0

    core_grid = ttnn.CoreGrid(y=shard_grid[1], x=shard_grid[0])
    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    logger.debug(f"shard_grid={shard_grid}")
    logger.debug(f"grid_size={core_grid}")
    logger.debug(f"input_shard_height={shard_height}, input_shard_width={shard_width}")

    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)

    logger.debug(f"shard_memory_layout={in_sharded_mem_config}")
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
    ##op computation
    if op == "silu":
        torch_silu = torch.nn.SiLU()
        torch_result = torch_silu(input)
        output_tensor = ttnn.silu(input_tensor, memory_config=in_sharded_mem_config)
    elif op == "relu":
        torch_relu = torch.nn.ReLU()
        torch_result = torch_relu(input)
        output_tensor = ttnn.relu(input_tensor, memory_config=in_sharded_mem_config)

    # output comparision
    output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(batch_size, input_height, input_width, input_channels)

    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)
    input = input.permute(0, 2, 3, 1)
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_result, output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, ncores, shard_grid, shard_strategy, shard_orientation",
    (
        (2, 320, 64, 64, 40, (8, 5), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR),
        (8, 256, 56, 56, 98, (12, 9), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
        (8, 512, 28, 28, 32, (4, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR),
        (8, 1024, 14, 14, 56, (7, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR),
        (16, 256, 56, 56, 98, (12, 9), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
    ),
)
@pytest.mark.parametrize("op", ["silu", "relu"])
def test_gs_silu_relu(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    shard_grid,
    ncores,
    shard_strategy,
    shard_orientation,
    op,
):
    run_elt_silu_relu(
        device,
        batch_size,
        input_channels,
        input_height,
        input_width,
        shard_grid,
        ncores,
        shard_strategy,
        shard_orientation,
        op,
    )


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, ncores, shard_grid, shard_strategy, shard_orientation",
    (
        (2, 320, 64, 64, 40, (8, 5), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR),
        (8, 256, 56, 56, 32, (8, 4), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
        (8, 512, 28, 28, 32, (4, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR),
        (8, 1024, 14, 14, 56, (7, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COLUMN_MAJOR),
        (8, 256, 56, 56, 56, (7, 8), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
    ),
)
@pytest.mark.parametrize("op", ["silu", "relu"])
def test_wh_silu_relu(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    shard_grid,
    ncores,
    shard_strategy,
    shard_orientation,
    op,
):
    run_elt_silu_relu(
        device,
        batch_size,
        input_channels,
        input_height,
        input_width,
        shard_grid,
        ncores,
        shard_strategy,
        shard_orientation,
        op,
    )
