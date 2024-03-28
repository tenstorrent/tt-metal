# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import numpy as np  # remove this
import tt_lib as ttl
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout, update_process_id
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import ttnn
from tt_lib.utils import (
    _nearest_y,
)


def write_to_file(tensor, file_name):
    tensor = tensor.cpu().numpy()
    print(tensor.shape)
    with open(file_name, "w") as f:
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    for l in range(tensor.shape[3]):
                        f.write(f"{tensor[i][j][k][l]} ")
                    f.write(f"\n")
                f.write(f"\n")


@pytest.mark.parametrize(
    "input_height, input_width, num_cores, shard_grid, shard_stragey",
    (
        (2048, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (8192, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (128, 1280, 40, (8, 5), ttnn.ShardStrategy.WIDTH),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "op",
    [ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpMath.MUL],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_bcast(
    device, use_program_cache, batch_size, input_height, input_width, num_cores, shard_grid, shard_stragey, dtype, op
):
    torch.manual_seed(0)
    input_shape = [batch_size, 1, input_height, input_width]
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    input_height = input_height * batch_size
    tt_input = input.reshape(1, 1, input_height, input_width)
    input_tensor = ttnn.from_torch(
        tt_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=dtype
    )
    input_2d_height = input_tensor.get_legacy_shape()[2]
    input_2d_width = input_tensor.get_legacy_shape()[3]
    if shard_stragey == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, shard_grid[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / shard_grid[0])
        shard_width = math.ceil(input_2d_width / shard_grid[1])
        shard_orientation = ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
        core_grid = ttnn.CoreGrid(y=shard_grid[1], x=shard_grid[0])
    else:
        shard_height = input_2d_height
        shard_width = math.ceil(input_2d_width / num_cores)
        shard_orientation = ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
        core_grid = get_shard_grid_from_num_cores(num_cores, device)

    logger.debug(f"core_grid={core_grid}")
    logger.debug(f"input_2d_height={input_2d_height} and input_2d_width={input_2d_width}")
    logger.debug(f"shard_height={shard_height} and shard_width={shard_width}")

    in_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_width, shard_height),
        core_grid=core_grid,
        strategy=shard_stragey,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    tt_input = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)

    b_weights_shape = [batch_size, 1, 1, input_width]
    B_pyt = torch.rand(size=b_weights_shape).bfloat16()
    if op == ttl.tensor.BcastOpMath.ADD:
        torch_ref_output = torch.add(input, B_pyt)
    elif op == ttl.tensor.BcastOpMath.MUL:
        torch_ref_output = torch.mul(input, B_pyt)

    B_pyt = B_pyt.reshape(b_weights_shape)
    tt_weight = ttnn.from_torch(B_pyt, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    tt_output = ttl.tensor.bcast(
        tt_input,
        tt_weight,
        op,
        ttl.tensor.BcastOpDim.H,
        output_mem_config=ttnn.get_memory_config(tt_input),
    )

    output_tensor = ttnn.to_torch(tt_output).float()
    output_tensor = output_tensor.reshape(input_shape)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_ref_output, output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing
