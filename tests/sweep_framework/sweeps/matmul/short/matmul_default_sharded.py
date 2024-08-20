# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger
import enum

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


class TensorMemoryConfigs(enum.Enum):
    BLOCK_Y5_X7 = enum.auto()
    BLOCK_Y7_X5_COL = enum.auto()
    BLOCK_Y2_X2 = enum.auto()
    WIDTH_Y1_X7 = enum.auto()
    HEIGHT_Y7_X1 = enum.auto()
    HEIGHT_Y7_X7_USE_H_W = enum.auto()


parameters = {
    "mcast_2d": {
        "batch_sizes": [(2, 3)],
        "input_shapes": [(1600, 224, 896)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_memory_config_specs": [TensorMemoryConfigs.BLOCK_Y5_X7.name],
        "input_b_sharded_memory_config_specs": [None],
    },
    "mcast_2d_transposed": {
        "batch_sizes": [(2, 3)],
        "input_shapes": [(1600, 224, 896)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_memory_config_specs": [TensorMemoryConfigs.BLOCK_Y7_X5_COL.name],
        "input_b_sharded_memory_config_specs": [None],
    },
    "mcast_2d_shard_width_gt_1_TILE": {
        "batch_sizes": [(2, 1)],
        "input_shapes": [(128, 256, 512)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_memory_config_specs": [TensorMemoryConfigs.BLOCK_Y2_X2.name],
        "input_b_sharded_memory_config_specs": [None],
    },
    "mcast_in0": {
        "batch_sizes": [(2, 3)],
        "input_shapes": [(64, 32 * 7, 1024)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_memory_config_specs": [TensorMemoryConfigs.WIDTH_Y1_X7.name],
        "input_b_sharded_memory_config_specs": [None],
    },
    "mcast_in1": {
        "batch_sizes": [(2, 3)],
        "input_shapes": [(160 * 7, 64, 64)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_memory_config_specs": [TensorMemoryConfigs.HEIGHT_Y7_X1.name],
        "input_b_sharded_memory_config_specs": [None],
    },
    "bmm": {
        "batch_sizes": [(7, 7)],
        "input_shapes": [(384, 64, 384)],
        "batch_matrix_multiply": [True],
        "input_a_sharded_memory_config_specs": [TensorMemoryConfigs.HEIGHT_Y7_X7_USE_H_W.name],
        "input_b_sharded_memory_config_specs": [TensorMemoryConfigs.HEIGHT_Y7_X7_USE_H_W.name],
    },
}
# Add the rest of the parameters.
general = {
    "compute_kernel_config": [None],
    "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_layout": [ttnn.TILE_LAYOUT],
}
for p in parameters.values():
    p.update(general)


def get_config_dict(config):
    if config == TensorMemoryConfigs.BLOCK_Y5_X7.name:
        return dict(core_grid=ttnn.CoreGrid(y=5, x=7), strategy=ttnn.ShardStrategy.BLOCK)
    elif config == TensorMemoryConfigs.BLOCK_Y7_X5_COL.name:
        return dict(
            core_grid=ttnn.CoreGrid(y=7, x=5),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        )
    elif config == TensorMemoryConfigs.BLOCK_Y2_X2.name:
        return dict(core_grid=ttnn.CoreGrid(y=2, x=2), strategy=ttnn.ShardStrategy.BLOCK)
    elif config == TensorMemoryConfigs.WIDTH_Y1_X7.name:
        return dict(
            core_grid=ttnn.CoreGrid(y=1, x=7),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
    elif config == TensorMemoryConfigs.HEIGHT_Y7_X1.name:
        return dict(
            core_grid=ttnn.CoreGrid(y=7, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
        )
    elif config == TensorMemoryConfigs.HEIGHT_Y7_X7_USE_H_W.name:
        return dict(
            core_grid=ttnn.CoreGrid(y=7, x=7),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        raise Exception(f"config {config} is not a supported enum.")


def run(
    batch_sizes,
    input_shapes,
    batch_matrix_multiply,
    input_a_sharded_memory_config_specs,
    input_b_sharded_memory_config_specs,
    compute_kernel_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    *,
    device,
) -> list:
    (m_size, k_size, n_size) = input_shapes
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

    input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG
    input_b_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if input_a_sharded_memory_config_specs:
        input_a_memory_config = ttnn.create_sharded_memory_config(
            input_shape_a, **get_config_dict(input_a_sharded_memory_config_specs)
        )
    if input_b_sharded_memory_config_specs:
        input_b_memory_config = ttnn.create_sharded_memory_config(
            input_shape_b, **get_config_dict(input_b_sharded_memory_config_specs)
        )

    input_a_layout = input_layout
    input_b_layout = input_layout

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=input_a_layout,
        dtype=input_a_dtype,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=input_b_layout,
        dtype=input_b_dtype,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=output_memory_config, dtype=output_dtype)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    expected_pcc = 0.99
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
