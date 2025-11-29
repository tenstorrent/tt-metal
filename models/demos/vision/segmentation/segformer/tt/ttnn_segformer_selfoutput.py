# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtSegformerSelfOutput:
    def __init__(self):
        super().__init__()

    def __call__(self, device, hidden_states: ttnn.Tensor, parameters):
        if len(hidden_states.shape) == 4:
            batch_size, __, seq_len, hidden_size = hidden_states.shape
        elif len(hidden_states.shape) == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape

        mm_f_x_strategy = ttnn.ShardStrategy.HEIGHT
        mm_f_x_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        mm_f_y = 8
        if (seq_len == 256) and (hidden_size == 256):
            mm_f_x = 8
            mm_f_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_f_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (seq_len == 1024) and (hidden_size == 160):
            mm_f_x = 5
            mm_f_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_f_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (seq_len == 4096) and (hidden_size == 64):
            mm_f_x = 8
        elif (seq_len == 16384) and (hidden_size == 32):
            mm_f_x = 8

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
            parameters.dense.weight,
            bias=parameters.dense.bias,
            memory_config=mm_f_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_f_y, x=mm_f_x),
            dtype=ttnn.bfloat8_b,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        return hidden_states
