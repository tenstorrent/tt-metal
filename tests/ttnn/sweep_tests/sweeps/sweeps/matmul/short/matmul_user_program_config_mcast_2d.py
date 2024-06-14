# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger
import enum

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, assert_with_pcc
from models.utility_functions import torch_random


class TensorMemoryConfigs(enum.Enum):
    CUSTOM_MEMORY_CONFIG = enum.auto()


parameters = {
    "matmul_specs": [
        # Matmul 2D mcast in0: in0 grid == output grid along tensor width
        (
            (1,),
            (5 * 128, 7 * 64, 7 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 5),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 4))}),
                    (128, 64),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 2D mcast in0: in0 grid < output grid along tensor width
        (
            (1,),
            (5 * 128, 4 * 64, 7 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 5),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 4))}),
                    (128, 64),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 2D mcast in0: in0 grid > output grid along tensor width
        (
            (1,),
            (5 * 128, 7 * 64, 4 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 5),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,  # Modified by transpose_mcast test parameter
                fused_activation=None,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 4))}),
                    (128, 64),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
    ],
    "batch_matrix_multiply": [False],
    "in0_block_w": [
        1,
        2,
    ],  # Used to override in0_block_w in program config (1: loop along in0 shard width; 2: no looping along in0 shard width)
    "transpose_mcast": [False, True],  # Used to override transpose_mcast in program config
    "input_a_memory_config": [TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
    "input_a_dtype": [ttnn.bfloat16],
    "input_b_dtype": [ttnn.bfloat8_b],
    "output_dtype": [ttnn.bfloat16],
    "input_layout": [ttnn.TILE_LAYOUT],
    "compute_kernel_config": [None],
}


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    matmul_specs,
    batch_matrix_multiply,
    in0_block_w,
    transpose_mcast,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    compute_kernel_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    (
        batch_sizes,
        input_shapes,
        program_config,
        input_a_custom_memory_config,
    ) = matmul_specs

    program_config.in0_block_w = in0_block_w
    program_config.transpose_mcast = transpose_mcast
    if program_config.transpose_mcast:
        input_a_shard_grid_end = input_a_custom_memory_config.shard_spec.grid.bounding_box().end
        input_a_custom_memory_config.shard_spec.grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(input_a_shard_grid_end.y, input_a_shard_grid_end.x))}
        )
        input_a_custom_memory_config.shard_spec.orientation = ttnn.ShardOrientation.COL_MAJOR

        compute_with_storage_grid_size = program_config.compute_with_storage_grid_size
        program_config.compute_with_storage_grid_size = (
            compute_with_storage_grid_size.y,
            compute_with_storage_grid_size.x,
        )

    if input_a_memory_config == TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG:
        input_a_memory_config = input_a_custom_memory_config

    (m_size, k_size, n_size) = input_shapes
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

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

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_memory_config,
        dtype=output_dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    expected_pcc = 0.99
    return check_with_pcc(torch_output_tensor, output_tensor, expected_pcc)
