# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import numpy as np  # remove this
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout, update_process_id
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from models.utility_functions import skip_for_blackhole
import ttnn
from tt_lib.utils import (
    _nearest_y,
)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize(
    "input_height, input_width, num_cores, shard_grid, shard_strategy",
    (
        (2048, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (128, 1280, 40, (8, 5), ttnn.ShardStrategy.WIDTH),
        (8192, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (128, 1280, 32, (4, 8), ttnn.ShardStrategy.BLOCK),
        (512, 1280, 64, (8, 8), ttnn.ShardStrategy.BLOCK),
    ),
)
@pytest.mark.parametrize(
    "in0_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "in1_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "op",
    [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.MUL],
)
@pytest.mark.parametrize("in1_batch_size", [1, 2])
@pytest.mark.parametrize("in0_batch_size", [1, 2])
@pytest.mark.parametrize(
    "orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_bcast(
    device,
    use_program_cache,
    orientation,
    in0_batch_size,
    in1_batch_size,
    input_height,
    input_width,
    num_cores,
    shard_grid,
    shard_strategy,
    in0_dtype,
    in1_dtype,
    op,
):
    torch.manual_seed(0)
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        if shard_strategy == ttnn.ShardStrategy.BLOCK:
            shard_grid = (
                (shard_grid[0], 4)
                if shard_grid[1] == 8 and orientation == ttnn.ShardOrientation.COL_MAJOR
                else shard_grid
            )
            shard_grid = (
                (4, shard_grid[1])
                if shard_grid[0] == 8 and orientation == ttnn.ShardOrientation.ROW_MAJOR
                else shard_grid
            )
    input_shape = [in0_batch_size, 1, input_height, input_width]
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=in0_dtype
    )
    input_2d_height = (
        input_tensor.shape.with_tile_padding()[0]
        * input_tensor.shape.with_tile_padding()[1]
        * input_tensor.shape.with_tile_padding()[2]
    )
    input_2d_width = input_tensor.shape.with_tile_padding()[3]
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, shard_grid[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / shard_grid[0])
        shard_width = math.ceil(input_2d_width / shard_grid[1])
        shard_orientation = orientation
        core_grid = (
            ttnn.CoreGrid(y=shard_grid[0], x=shard_grid[1])
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR
            else ttnn.CoreGrid(y=shard_grid[1], x=shard_grid[0])
        )
    else:
        shard_height = input_2d_height
        shard_width = math.ceil(input_2d_width / num_cores)
        shard_orientation = orientation
        core_grid = get_shard_grid_from_num_cores(num_cores, device)

    logger.debug(f"core_grid={core_grid}")
    logger.debug(f"input_2d_height={input_2d_height} and input_2d_width={input_2d_width}")
    logger.debug(f"shard_height={shard_height} and shard_width={shard_width}")

    in_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(
            (shard_height, shard_width)
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR
            else (shard_width, shard_height)
        ),
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    tt_input = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)

    if in0_batch_size == 1 and in1_batch_size > 1:
        input = input.reshape(in1_batch_size, 1, input_height // in1_batch_size, input_width)

    b_weights_shape = [in1_batch_size, 1, 1, input_width]
    B_pyt = torch.rand(size=b_weights_shape).bfloat16()
    if op == ttnn.BcastOpMath.ADD:
        torch_ref_output = torch.add(input, B_pyt)
    elif op == ttnn.BcastOpMath.MUL:
        torch_ref_output = torch.mul(input, B_pyt)

    if in0_batch_size == 1 and in1_batch_size > 1:
        torch_ref_output = torch_ref_output.reshape(1, 1, input_height, input_width)

    B_pyt = B_pyt.reshape(b_weights_shape)
    tt_weight = ttnn.from_torch(B_pyt, device=device, layout=ttnn.TILE_LAYOUT, dtype=in1_dtype)
    tt_output = ttnn.bcast(
        tt_input,
        tt_weight,
        op,
        ttnn.BcastOpDim.H,
        memory_config=ttnn.get_memory_config(tt_input),
    )

    output_tensor = ttnn.to_torch(tt_output).float()
    output_tensor = output_tensor.reshape(input_shape)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_ref_output, output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing
