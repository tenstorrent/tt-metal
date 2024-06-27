# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from ttnn.dot_access import DotAccessDict


def update_model_config(config, batch_size):
    core_grid = ttnn.CoreGrid(y=8, x=12)

    program_configs = {
        "fold_output_program_config": ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            ttnn.experimental.tensor.ShardSpec(
                ttnn.experimental.tensor.CoreRangeSet(
                    {
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, 0),
                            ttnn.experimental.tensor.CoreCoord(12, 7),
                        ),
                    }
                ),
                [
                    224,
                    192,
                ],
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        ),
        "embedding_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=3,
            out_subblock_h=1,
            out_subblock_w=6,
            per_core_M=7,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            per_core_M=7,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=7,
            per_core_M=7,
            per_core_N=7,
        ),
        "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=7,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=7,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=7,
            per_core_N=8,
            transpose_mcast=False,
            fused_activation=(ttnn.experimental.tensor.FusibleActivation.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=8,
            out_subblock_h=7,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "classifer_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            per_core_M=7,
            per_core_N=3,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "layernorm_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=2,
            block_h=7,
            block_w=2,
            # math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
            # im_data_format=ttnn.experimental.tensor.DataType.BFLOAT16,
            # out_data_format=ttnn.bfloat8_b,
            inplace=False,
        ),
        "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=2,
            block_h=7,
            block_w=2,
            # math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
            # im_data_format=ttnn.experimental.tensor.DataType.BFLOAT16,
            # out_data_format=ttnn.bfloat8_b,
            inplace=False,
        ),
        "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=1,
            block_h=7,
            block_w=7,
            # math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
            # im_data_format=ttnn.experimental.tensor.DataType.BFLOAT16,
        ),
    }

    return DotAccessDict(dict(**config.to_dict(), core_grid=core_grid, program_configs=program_configs))
