# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

program_configs = {
    "linear_config_4096": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=4,
        per_core_N=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_config_16384": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=16,
        per_core_N=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_config_1024": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(5, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=4,
        per_core_N=8,
        transpose_mcast=False,
        fused_activation=None,
    ),
    "linear_config_256": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=2,
        transpose_mcast=False,
        fused_activation=None,
    ),
}


class TtSegformerMLP:
    def __init__(self):
        super().__init__()

    def __call__(self, device, hidden_states: ttnn.Tensor, parameters):
        mm_f_x_strategy = ttnn.ShardStrategy.HEIGHT
        mm_f_x_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        mm_f_y = 8
        if (hidden_states.shape[-2] == 256) and (hidden_states.shape[-1] == 256):
            mm_f_x = 4
            mm_f_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_f_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            mm_prog_config = program_configs["linear_config_256"]
        elif (hidden_states.shape[-2] == 1024) and (hidden_states.shape[-1] == 160):
            mm_f_x = 5
            mm_f_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_f_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            mm_prog_config = program_configs["linear_config_1024"]
        elif (hidden_states.shape[-2] == 4096) and (hidden_states.shape[-1] == 64):
            mm_f_x = 4
            mm_prog_config = program_configs["linear_config_4096"]
        elif (hidden_states.shape[-2] == 16384) and (hidden_states.shape[-1] == 32):
            mm_f_x = 4
            mm_prog_config = program_configs["linear_config_16384"]

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        if (hidden_states.shape[-2] == 1024) and (hidden_states.shape[-1] == 160):
            # TODO: convert it to sharding
            hidden_states = ttnn.linear(
                hidden_states,
                parameters.proj.weight,
                bias=parameters.proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
            )
        else:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_f_y, x=mm_f_x),
                    strategy=mm_f_x_strategy,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
            hidden_states = ttnn.linear(
                hidden_states,
                parameters.proj.weight,
                bias=parameters.proj.bias,
                memory_config=mm_f_x_memory_config,
                program_config=mm_prog_config,
                dtype=ttnn.bfloat8_b,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        return hidden_states
