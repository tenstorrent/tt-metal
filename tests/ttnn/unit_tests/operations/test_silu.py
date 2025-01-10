# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from typing import Union, Tuple
from loguru import logger
import torch

# import torch.nn as nn

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from models.utility_functions import is_wormhole_b0, skip_for_grayskull, is_blackhole
from tt_lib.utils import (
    _nearest_y,
)

TILE_WIDTH = 32


def run_elt_silu_relu(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    grid_size,
    ncores,
    shard_strategy,
    shard_orientation,
    op,
    dtype=ttnn.bfloat16,
):
    ## input shape is N C H W
    input_shape = [batch_size, input_channels, input_height, input_width]
    torch.manual_seed(0)
    input = torch.rand(input_shape, dtype=torch.bfloat16) * 2 - 1

    ## permute to N H W C, which is what the upsample op expects
    tt_input = input.permute(0, 2, 3, 1)
    tt_input = tt_input.reshape(1, 1, batch_size * input_height * input_width, input_channels)
    input_tensor = ttnn.from_torch(
        tt_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=dtype
    )
    interleaved_mem_config = ttnn.L1_MEMORY_CONFIG
    input_tensor = ttnn.to_memory_config(input_tensor, interleaved_mem_config)
    # input_shape = [1, 1, _nearest_y(batch_size * input_height * input_width, 32), input_channels]
    input_2d_height = input_tensor.shape.with_tile_padding()[2]
    input_2d_width = input_tensor.shape.with_tile_padding()[3]
    logger.debug(f"input_2d_height={input_2d_height} and input_2d_width={input_2d_width}")

    ## input shard
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, grid_size[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / grid_size[0])
        shard_width = math.ceil(input_2d_width / grid_size[1])
        shard_orientation = ttnn.ShardOrientation.COL_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        input_2d_height_padded = _nearest_y(input_2d_height, ncores * 32)
        shard_height = math.ceil(input_2d_height_padded / ncores)
        shard_grid = get_shard_grid_from_num_cores(ncores, device)
        shard_width = input_2d_width
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        shard_height = input_2d_height
        input_2d_width_padded = _nearest_y(input_2d_width, ncores * 32)
        shard_width = math.ceil(input_2d_width_padded / ncores)
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        shard_grid = get_shard_grid_from_num_cores(ncores, device)

    assert shard_height % TILE_WIDTH == 0
    assert shard_width % TILE_WIDTH == 0

    logger.debug(f"shard_grid={shard_grid}")
    logger.debug(f"input_shard_height={shard_height}, input_shard_width={shard_width}")

    shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

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


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, ncores, grid_size, shard_strategy, shard_orientation",
    (
        (2, 320, 64, 64, 40, (8, 5), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR),
        (8, 256, 56, 56, 98, (12, 9), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
        (8, 512, 28, 28, 32, (4, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR),
        (8, 1024, 14, 14, 56, (7, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR),
        (16, 256, 56, 56, 98, (12, 9), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
        (1, 5120, 32, 1, 32, (1, 32), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR),
        (1, 1024, 64, 1, 32, (1, 32), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR),
        (2, 10240, 64, 1, 64, (1, 64), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR),
    ),
)
@pytest.mark.parametrize("op", ["silu", "relu"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_gs_silu_relu(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    grid_size,
    ncores,
    shard_strategy,
    shard_orientation,
    op,
    dtype,
):
    run_elt_silu_relu(
        device,
        batch_size,
        input_channels,
        input_height,
        input_width,
        grid_size,
        ncores,
        shard_strategy,
        shard_orientation,
        op,
        dtype,
    )


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, ncores, grid_size, shard_strategy, shard_orientation",
    (
        (2, 320, 64, 64, 40, (8, 5), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR),
        (8, 256, 56, 56, 32, (8, 4), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
        (8, 512, 28, 28, 32, (4, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR),
        (8, 1024, 14, 14, 56, (7, 8), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR),
        (8, 256, 56, 56, 56, (7, 8), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR),
        (1, 5120, 32, 1, 32, (1, 32), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR),
        (1, 1024, 64, 1, 32, (1, 32), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR),
        (2, 10240, 64, 1, 32, (1, 32), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR),
    ),
)
@pytest.mark.parametrize("op", ["silu", "relu"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_wh_silu_relu(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    grid_size,
    ncores,
    shard_strategy,
    shard_orientation,
    op,
    dtype,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        if shard_strategy == ttnn.ShardStrategy.BLOCK:
            grid_size = (grid_size[0], 4) if grid_size[1] == 8 else grid_size  # reduce #f core used for N300

    run_elt_silu_relu(
        device,
        batch_size,
        input_channels,
        input_height,
        input_width,
        grid_size,
        ncores,
        shard_strategy,
        shard_orientation,
        op,
        dtype,
    )


@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, ncores, grid_size, shard_strategy, shard_orientation",
    ((1, 5120, 32, 1, 32, (1, 32), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR),),
)
@pytest.mark.parametrize("op", ["silu", "relu"])
def test_silu_llm(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    grid_size,
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
        grid_size,
        ncores,
        shard_strategy,
        shard_orientation,
        op,
    )
