# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger
import enum

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Set up suites and specify batch_sizes, input_shapes, batch_matrix_multiply, program_config, and memory configs
# TODO: Missing coverage for WH fp32 and compute config tests in: tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul.py
# [1d in0 batched, bmm no mcast per_core_M > M, bmm no mcast per_core_M < M, 1d mcast in1 with height padding, 1d mcast in1, 1d mcast in0, 2d mcast, 2d mcast transposed, 2d mcast height/width sharded] x [specific L1 memory, dram memory] x [a=bfloat16, a=bfloat8_b] x [b=bfloat16, b=bfloat8_b] x [out=bfloat16, out=bfloat8_b]
# 144 tests.
# Use the first two categories for suites: 18 suites with 8 tests each.
parameters = {
    "matmul_1d_in0_batched": {
        "batch_sizes": [(16,)],
        "input_shapes": [(128, 128, 1024)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_MEMORY_CONFIG],
    },
    "matmul_1d_in0_batched_dram": {
        "batch_sizes": [(16,)],
        "input_shapes": [(128, 128, 1024)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "bmm_no_mcast_per_core_M_gt_M": {
        "batch_sizes": [(2, 16)],
        "input_shapes": [(384, 64, 128)],
        "batch_matrix_multiply": [True],
        "program_config": [
            ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=12,  #  B * H * M // num_cores(32) // 32
                per_core_N=4,  # N // 32
            )
        ],
        "input_a_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (384, 64),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
        "input_b_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (64, 128),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
    },
    "bmm_no_mcast_per_core_M_gt_M_dram": {
        "batch_sizes": [(2, 16)],
        "input_shapes": [(384, 64, 128)],
        "batch_matrix_multiply": [True],
        "program_config": [
            ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=12,  #  B * H * M // num_cores(32) // 32
                per_core_N=4,  # N // 32
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "bmm_no_mcast_per_core_M_lt_M": {
        "batch_sizes": [(4, 1)],
        "input_shapes": [(1024, 64, 1024)],
        "batch_matrix_multiply": [True],
        "program_config": [
            ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=4,  #  B * H * M // num_cores(32) // 32
                per_core_N=32,  # N // 32
            )
        ],
        "input_a_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (128, 64),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
        "input_b_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
    },
    "bmm_no_mcast_per_core_M_lt_M_dram": {
        "batch_sizes": [(4, 1)],
        "input_shapes": [(1024, 64, 1024)],
        "batch_matrix_multiply": [True],
        "program_config": [
            ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=4,  #  B * H * M // num_cores(32) // 32
                per_core_N=32,  # N // 32
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "1d_mcast_in1_height_padded": {
        "batch_sizes": [(1,)],
        "input_shapes": [(63 * 32, 32, 32)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (2 * 32, 32),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
    },
    "1d_mcast_in1_height_padded_dram": {
        "batch_sizes": [(1,)],
        "input_shapes": [(63 * 32, 32, 32)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "1d_mcast_in1": {
        "batch_sizes": [(1,)],
        "input_shapes": [(8704, 64, 64)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [
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
            )
        ],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
    },
    "1d_mcast_in1_dram": {
        "batch_sizes": [(1,)],
        "input_shapes": [(8704, 64, 64)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "1d_mcast_in0": {
        "batch_sizes": [(1,)],
        "input_shapes": [(64, 2048, 1024)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (64, 64),  # M, K // num_cores(32)
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
        "input_b_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG],
    },
    "1d_mcast_in0_dram": {
        "batch_sizes": [(1,)],
        "input_shapes": [(64, 2048, 1024)],
        "batch_matrix_multiply": [False],
        "program_config": [
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
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "2d_mcast": {
        "batch_sizes": [(1,)],
        "input_shapes": [(1600, 512, 1024)],
        "batch_matrix_multiply": [False],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 5),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=10,  # M // grid_size[1] // 32
                per_core_N=4,  # N // grid_size[0] // 32
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
                    (320, 64),  # M // grid_size[1], K // grid_size[0]
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
        "input_b_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
    },
    "2d_mcast_dram": {
        "batch_sizes": [(1,)],
        "input_shapes": [(1600, 512, 1024)],
        "batch_matrix_multiply": [False],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 5),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=10,  # M // grid_size[1] // 32
                per_core_N=4,  # N // grid_size[0] // 32
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "2d_mcast_transposed": {
        "batch_sizes": [(1,)],
        "input_shapes": [(1600, 256, 512)],
        "batch_matrix_multiply": [False],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=10,  # M // grid_size[0] // 32
                per_core_N=4,  # N // grid_size[1] // 32
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
                    (320, 64),  # M // grid_size[0], K // grid_size[1]
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
        "input_b_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
    },
    "2d_mcast_transposed_dram": {
        "batch_sizes": [(1,)],
        "input_shapes": [(1600, 256, 512)],
        "batch_matrix_multiply": [False],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=10,  # M // grid_size[0] // 32
                per_core_N=4,  # N // grid_size[1] // 32
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "2d_mcast_in0_height_sharded_in1_width_sharded": {
        "batch_sizes": [(1,)],
        "input_shapes": [(192, 64, 384)],
        "batch_matrix_multiply": [False],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(6, 6),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,  # M // grid_size[1] // 32
                per_core_N=2,  # N // grid_size[0] // 32
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 5))}),
                    (32, 64),  # M // grid_size[1], K
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
        "input_b_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
                    (64, 64),  # K, N // grid_size[0]
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
        "output_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
    },
    "2d_mcast_in0_in1_dram": {
        "batch_sizes": [(1,)],
        "input_shapes": [(192, 64, 384)],
        "batch_matrix_multiply": [False],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(6, 6),
                in0_block_w=2,  # K // 32
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,  # M // grid_size[1] // 32
                per_core_N=2,  # N // grid_size[0] // 32
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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


def run(
    batch_sizes,
    input_shapes,
    batch_matrix_multiply,
    program_config,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
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
    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_memory_config,
        dtype=output_dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    expected_pcc = 0.99
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
