# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
}


class TtSegformerMLP:
    def __init__(self):
        super().__init__()

    def __call__(self, hidden_states: ttnn.Tensor, parameters):
        device = hidden_states.device()
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(
            hidden_states,
            (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2] * hidden_states.shape[3]),
        )
        hidden_states = ttnn.to_device(hidden_states, device=device)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        if len(hidden_states.shape) == 2:  # This is due to while permuting 1,x,y we are getting 2D as output
            hidden_states = ttnn.reshape(hidden_states, (1, hidden_states.shape[0], hidden_states.shape[1]))

        if hidden_states.shape[1] > 4096:
            mm_1_y = 8
            mm_1_x = 4
        elif hidden_states.shape[1] == 4096:
            mm_1_y = 8
            mm_1_x = 4
        elif hidden_states.shape[1] == 1024:
            mm_1_y = 8
            mm_1_x = 5
        elif hidden_states.shape[1] == 256:
            mm_1_y = 8
            mm_1_x = 4

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        print("---mlp MM----", hidden_states.shape)

        if hidden_states.shape[1] > 4096:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
            hidden_states = ttnn.linear(
                hidden_states,
                parameters.proj.weight,
                bias=parameters.proj.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                # core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                program_config=program_configs["linear_config_16384"],
                dtype=ttnn.bfloat8_b,
            )
        elif hidden_states.shape[1] == 4096:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
            hidden_states = ttnn.linear(
                hidden_states,
                parameters.proj.weight,
                bias=parameters.proj.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                program_config=program_configs["linear_config_4096"],
            )
        elif hidden_states.shape[1] == 1024:
            # hidden_states = ttnn.to_memory_config(
            #     hidden_states,
            #     memory_config=ttnn.create_sharded_memory_config(
            #         hidden_states.shape,
            #         core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
            #         strategy=ttnn.ShardStrategy.BLOCK,
            #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
            #     ),
            #     dtype=ttnn.bfloat8_b,
            # )

            # hidden_states = ttnn.linear(
            #     hidden_states,
            #      parameters.proj.weight,
            #     bias=parameters.proj.bias,
            #     memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            #     # core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
            #     program_config=program_configs["linear_config_1024"],
            #     dtype=ttnn.bfloat8_b,
            # )

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
        elif hidden_states.shape[1] == 256:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.proj.weight,
                bias=parameters.proj.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                dtype=ttnn.bfloat8_b,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # hidden_states = ttnn.linear(
        #     hidden_states,
        #     parameters.proj.weight,
        #     bias=parameters.proj.bias,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     core_grid=ttnn.CoreGrid(y=8, x=8),
        #     compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        #         math_fidelity=ttnn.MathFidelity.LoFi,
        #     ),
        # )
        return hidden_states
