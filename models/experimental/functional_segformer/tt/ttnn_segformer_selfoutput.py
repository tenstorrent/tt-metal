# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtSegformerSelfOutput:
    def __init__(self):
        super().__init__()

    def __call__(self, hidden_states: ttnn.Tensor, parameters):
        print("mm-6--", hidden_states.shape)

        # hidden_states = ttnn.linear(
        #     hidden_states,
        #     parameters.dense.weight,
        #     bias=parameters.dense.bias,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     core_grid=ttnn.CoreGrid(y=8, x=8),
        #     # dtype=ttnn.bfloat8_b, Hangs for some inputs
        # )

        if hidden_states.shape[1] >= 2048:
            mm_6_y = 8
            mm_6_x = 4
        elif hidden_states.shape[1] == 1024:
            mm_6_y = 8
            mm_6_x = 5
        elif hidden_states.shape[1] == 256:
            mm_6_y = 8
            mm_6_x = 4

        if hidden_states.shape[1] >= 2048:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_6_y, x=mm_6_x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense.weight,
                bias=parameters.dense.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_6_y, x=mm_6_x),
                dtype=ttnn.bfloat8_b,
            )
        elif hidden_states.shape[1] == 1024:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_6_y, x=mm_6_x),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense.weight,
                bias=parameters.dense.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_6_y, x=mm_6_x),
                dtype=ttnn.bfloat8_b,
            )

        elif hidden_states.shape[1] == 256:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_6_y, x=mm_6_x),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense.weight,
                bias=parameters.dense.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_6_y, x=mm_6_x),
                dtype=ttnn.bfloat8_b,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        return hidden_states
