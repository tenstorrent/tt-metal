# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_dwconv import TtSegformerDWConv


class TtSegformerMixFFN:
    def __init__(self, parameters, hidden_features):
        super().__init__()
        self.dwconv = TtSegformerDWConv(parameters["dwconv"], hidden_features)

    def __call__(self, hidden_states: ttnn.Tensor, height: int, width: int, parameters, device):
        # print("mm-7--", hidden_states.shape, parameters.dense1.weight.shape)

        mm_f_x_strategy = ttnn.ShardStrategy.HEIGHT
        mm_f_x_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        mm_f_y = 8
        if (hidden_states.shape[-2] == 256) and (hidden_states.shape[-1] == 256):
            mm_f_x = 4
            mm_f_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_f_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (hidden_states.shape[-2] == 1024) and (hidden_states.shape[-1] == 160):
            mm_f_x = 5
            mm_f_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_f_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (hidden_states.shape[-2] == 4096) and (hidden_states.shape[-1] == 64):
            mm_f_x = 2
            mm_f_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_f_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (hidden_states.shape[-2] == 16384) and (hidden_states.shape[-1] == 32):
            mm_f_x = 4

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

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
            parameters.dense1.weight,
            bias=parameters.dense1.bias,
            memory_config=mm_f_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_f_y, x=mm_f_x),
            dtype=ttnn.bfloat8_b,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        hidden_states, __, __ = self.dwconv(hidden_states, height, width, device)
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        # TODO: GeLU on sharded data
        hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        # print("mm-8--", hidden_states.shape, parameters.dense2.weight.shape)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

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
            parameters.dense2.weight,
            bias=parameters.dense2.bias,
            memory_config=mm_f_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_f_y, x=mm_f_x),
            dtype=ttnn.bfloat8_b,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return hidden_states
