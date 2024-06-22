# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger
import enum

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


class TensorMemoryConfigs(enum.Enum):
    CUSTOM_MEMORY_CONFIG = enum.auto()


# TODO: Missing coverage for WH fp32 and compute config tests in: tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul.py
parameters = {
    "matmul_specs": [
        # Matmul 1D in0 batched
        (
            (16,),
            (128, 128, 1024),
            False,
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,  # M // 32
                per_core_N=1,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
        # BMM no mcast: per_core_M > M
        (
            (2, 16),
            (384, 64, 128),
            True,
            ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=12,  #  B * H * M // num_cores(32) // 32
                per_core_N=4,  # N // 32
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (384, 64),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (64, 128),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            ),
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ),
        # BMM no mcast: per_core_M < M
        (
            (4, 1),
            (1024, 64, 1024),
            True,
            ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=4,  #  B * H * M // num_cores(32) // 32
                per_core_N=32,  # N // 32
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (128, 64),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            ),
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ),
        # Matmul 1D mcast in1 with height padding
        (
            (1,),
            (63 * 32, 32, 32),
            False,
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=1,  # K // 32
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (2 * 32, 32),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ),
        # Matmul 1D mcast in1
        (
            (1,),
            (8704, 64, 64),
            False,
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 5),
                in0_block_w=2,  # K // 32
                out_subblock_h=4,  # 8 // (N // 32)
                out_subblock_w=2,  # N // 32
                per_core_M=8,  # M // num_cores(34) // 32
                per_core_N=2,  # N // 32
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(1, 4)),
                        }
                    ),
                    (256, 64),  # M // num_cores(34), K
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ),
        # Matmul 1D mcast in0
        (
            (1,),
            (64, 2048, 1024),
            False,
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,  # M // 32
                per_core_N=1,  # N // num_cores(32) // 32
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (64, 64),  # M, K // num_cores(32)
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ),
        # Matmul 2D mcast
        (
            (1,),
            (1600, 512, 1024),
            False,
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 5),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=10,  # M // grid_size[1] // 32
                per_core_N=4,  # N // grid_size[0] // 32
                transpose_mcast=False,
                fused_activation=None,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
                    (320, 64),  # M // grid_size[1], K // grid_size[0]
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ),
        # Matmul 2D mcast transposed
        (
            (1,),
            (1600, 256, 512),
            False,
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=10,  # M // grid_size[0] // 32
                per_core_N=4,  # N // grid_size[1] // 32
                transpose_mcast=True,
                fused_activation=None,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
                    (320, 64),  # M // grid_size[0], K // grid_size[1]
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            ),
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ),
        # Matmul 2D mcast in0 height sharded and in1 width sharded
        (
            (1,),
            (192, 64, 384),
            False,
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(6, 6),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,  # M // grid_size[1] // 32
                per_core_N=2,  # N // grid_size[0] // 32
                transpose_mcast=False,
                fused_activation=None,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 5))}),
                    (32, 64),  # M // grid_size[1], K
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
                    (64, 64),  # K, N // grid_size[0]
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ),
    ],
    "compute_kernel_config": [None],
    "tensor_memory_config": [TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
    "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_layout": [ttnn.TILE_LAYOUT],
}


def run(
    matmul_specs,
    compute_kernel_config,
    tensor_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    (
        batch_sizes,
        input_shapes,
        batch_matrix_multiply,
        program_config,
        input_a_custom_memory_config,
        input_b_custom_memory_config,
        output_custom_memory_config,
    ) = matmul_specs
    # TODO: This is to avoid looping through possible memory_configs in single runs; if we split sweep into separate ones, we can remove this hackiness
    if tensor_memory_config == TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG:
        input_a_memory_config = input_a_custom_memory_config
        input_b_memory_config = input_b_custom_memory_config
        output_memory_config = output_custom_memory_config
    else:
        input_a_memory_config = tensor_memory_config
        input_b_memory_config = tensor_memory_config
        output_memory_config = tensor_memory_config

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
