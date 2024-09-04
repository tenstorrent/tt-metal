# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger
import enum

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, assert_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 5


class TensorMemoryConfigs(enum.Enum):
    CUSTOM_MEMORY_CONFIG = enum.auto()
    DEFAULT_MEMORY_CONFIG = enum.auto()


IN0_INNER_DIM_PER_CORE = 96

# Set up suites and specify input_shapes, program_config, and input_a_custom_memory_config parameters.
parameters = {
    ########################################
    # TESTS: in0 inner dim != output width #
    ########################################
    # Matmul 2D mcast: in0 grid == output grid along tensor width
    "in0_grid_eq_output_grid": {
        "input_shapes": [(5 * 128, 7 * IN0_INNER_DIM_PER_CORE, 7 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 5),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 4))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
    "in0_grid_eq_output_grid_transpose": {
        "input_shapes": [(5 * 128, 7 * IN0_INNER_DIM_PER_CORE, 7 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 7),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 6))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
    },
    # Matmul 2D mcast: in0 grid < output grid along tensor width
    "in0_grid_lt_output_grid": {
        "input_shapes": [(5 * 128, 4 * IN0_INNER_DIM_PER_CORE, 7 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 5),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 4))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
    "in0_grid_lt_output_grid_transpose": {
        "input_shapes": [(5 * 128, 4 * IN0_INNER_DIM_PER_CORE, 7 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 7),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
    },
    # Matmul 2D mcast: in0 grid > output grid along tensor width
    "in0_grid_gt_output_grid": {
        "input_shapes": [(5 * 128, 7 * IN0_INNER_DIM_PER_CORE, 4 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 5),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 4))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
    "in0_grid_gt_output_grid_transpose": {
        "input_shapes": [(5 * 128, 7 * IN0_INNER_DIM_PER_CORE, 4 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 7),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 6))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
    },
    ###############################################################
    # TESTS: Single row/col or single core output grid edge cases #
    ###############################################################
    # Matmul 2D mcast in1 only: output grid along tensor width is 1
    "in1_only_output_grid_along_width_is_1": {
        "input_shapes": [(5 * 128, IN0_INNER_DIM_PER_CORE, 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(1, 5),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
    "in1_only_output_grid_along_width_is_1_transpose": {
        "input_shapes": [(5 * 128, IN0_INNER_DIM_PER_CORE, 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 1),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
    },
    # Matmul 2D mcast in0 only: output grid along tensor height is 1
    "in0_only_output_grid_along_height_is_1": {
        "input_shapes": [(128, 7 * IN0_INNER_DIM_PER_CORE, 4 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 1),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 0))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
    "in0_only_output_grid_along_height_is_1_transpose": {
        "input_shapes": [(128, 7 * IN0_INNER_DIM_PER_CORE, 4 * 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(1, 7),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
    },
    # Matmul 2D no mcast: output grid along tensor height and width are both 1
    "no_mcast_output_grid_is_1_by_1": {
        "input_shapes": [(128, IN0_INNER_DIM_PER_CORE, 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(1, 1),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
    "no_mcast_output_grid_is_1_by_1_transpose": {
        "input_shapes": [(128, IN0_INNER_DIM_PER_CORE, 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(1, 1),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
    },
    # Matmul 2D mcast in1 "only" special case: output grid along tensor width is 1 but in0 shard width > 1
    # in0 interleaved has no mcast but in0 sharded mcasts along tensor width to first core
    "in1_only_output_grid_width_1_in0_shard_width_gt_1": {
        "input_shapes": [(5 * 128, 7 * IN0_INNER_DIM_PER_CORE, 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 5),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 4))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
    "in1_only_output_grid_width_1_in0_shard_width_gt_1_transpose": {
        "input_shapes": [(5 * 128, 7 * IN0_INNER_DIM_PER_CORE, 96)],
        "program_config": [
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 7),
                in0_block_w=1,  # Modified by in0_block_w test parameter
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=4,
                per_core_N=3,
                transpose_mcast=True,
                fused_activation=None,
            )
        ],
        "input_a_custom_memory_config": [
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 6))}),
                    (128, IN0_INNER_DIM_PER_CORE),
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
        ],
    },
}

# Add the rest of the parameters.
general = {
    "batch_sizes": [(1,)],
    "batch_matrix_multiply": [False],
    "in0_block_w": [
        1,
        int(IN0_INNER_DIM_PER_CORE / 32),
    ],  # Used to override in0_block_w in program config (1: loop along in0 shard width; 2: no looping along in0 shard width)
    "input_a_memory_config": [TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG, TensorMemoryConfigs.DEFAULT_MEMORY_CONFIG],
    "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
    "input_a_dtype": [ttnn.bfloat16],
    "input_b_dtype": [ttnn.bfloat8_b],
    "output_dtype": [ttnn.bfloat16],
    "input_layout": [ttnn.TILE_LAYOUT],
    "compute_kernel_config": [None],
}
for p in parameters.values():
    p.update(general)


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    input_shapes,
    program_config,
    input_a_custom_memory_config,
    batch_matrix_multiply,
    in0_block_w,
    batch_sizes,
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
) -> list:
    # Override program_config.in0_block_w with value from in0_block_w (this is safe to do)
    program_config.in0_block_w = in0_block_w

    if input_a_memory_config == TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG:
        input_a_memory_config = input_a_custom_memory_config
    else:
        input_a_memory_config = ttnn.L1_MEMORY_CONFIG

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
