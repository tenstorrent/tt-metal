# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.segformer.tt.common import Conv


class TtSegformerEfficientSelfAttention:
    def __init__(self, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = Conv([sequence_reduction_ratio, sequence_reduction_ratio, 0, 0], parameters["sr"])

    def __call__(
        self,
        device,
        hidden_states: ttnn.Tensor,
        height: int,
        width: int,
        parameters,
        output_attentions=False,
    ):
        if len(hidden_states.shape) == 4:
            batch_size, __, seq_len, hidden_size = hidden_states.shape
        elif len(hidden_states.shape) == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape

        mm_a_x_strategy = ttnn.ShardStrategy.HEIGHT
        mm_a_x_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        mm_d_x_strategy = mm_a_x_strategy
        mm_d_x_memory_config = mm_a_x_memory_config
        mm_a_y = 8
        if (seq_len == 256) and (hidden_size == 256):
            mm_a_x = 8
            mm_b_x = 8
            mm_d_x = 2
            mm_d_y = 4
            mm_e_x = 8
            mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (seq_len == 1024) and (hidden_size == 160):
            mm_a_x = 5
            mm_b_x = 5
            mm_d_x = 5
            mm_d_y = 4
            mm_e_x = 8
            mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (seq_len == 4096) and (hidden_size == 64):
            mm_a_x = 8  # 8
            mm_b_x = 1  # 1
            mm_d_x = 4
            mm_d_y = 8
            mm_e_x = 8
        elif (seq_len == 16384) and (hidden_size == 32):
            mm_a_x = 8  # 8
            mm_b_x = 1  # 1
            mm_d_x = 8
            mm_d_y = 8
            mm_e_x = 8

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                strategy=mm_a_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat16,
        )
        query = ttnn.linear(
            hidden_states,
            parameters.query.weight,
            bias=parameters.query.bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            dtype=ttnn.bfloat16,
        )

        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        # Split Heads
        if self.num_attention_heads == 1:
            query_layer = query
        else:
            query_layer = ttnn.experimental.nlp_create_qkv_heads_segformer(query, memory_config=ttnn.L1_MEMORY_CONFIG)[
                0
            ]

        if self.sr_ratio > 1:
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, num_channels = hidden_states.shape
            elif len(hidden_states.shape) == 4:
                batch_size, __, seq_len, num_channels = hidden_states.shape

            # Need for RM input to reshape, then back to TILE after that
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, num_channels))
            if hidden_states.shape[3] == 160:
                # conv config update
                self.sr.output_layout = ttnn.TILE_LAYOUT
                hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

            hidden_states, __, __ = self.sr(device, hidden_states)
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm.weight,
                bias=parameters.layer_norm.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            )

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_b_x),
                strategy=mm_a_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        key = ttnn.linear(
            hidden_states,
            parameters.key.weight,
            bias=parameters.key.bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_b_x),
            dtype=ttnn.bfloat8_b,
        )

        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        if self.num_attention_heads == 1:
            key_layer = key
        else:
            key_layer = ttnn.experimental.nlp_create_qkv_heads_segformer(key, memory_config=ttnn.L1_MEMORY_CONFIG)[0]
        key_layer = ttnn.permute(key_layer, (0, 1, 3, 2))

        value = ttnn.linear(
            hidden_states,
            parameters.value.weight,
            bias=parameters.value.bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_b_x),
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(hidden_states)
        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        if self.num_attention_heads == 1:
            value_layer = value
        else:
            value_layer = ttnn.experimental.nlp_create_qkv_heads_segformer(value, memory_config=ttnn.L1_MEMORY_CONFIG)[
                0
            ]

        key_layer = ttnn.to_memory_config(key_layer, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        query_layer = ttnn.to_layout(query_layer, ttnn.TILE_LAYOUT)

        query_layer = ttnn.to_memory_config(
            query_layer,
            memory_config=ttnn.create_sharded_memory_config(
                query_layer.shape,
                core_grid=ttnn.CoreGrid(y=mm_d_y, x=mm_d_x),
                strategy=mm_d_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        attention_scores = ttnn.matmul(
            query_layer,
            key_layer,
            memory_config=mm_d_x_memory_config,
            dtype=ttnn.bfloat8_b,
        )

        ttnn.deallocate(query_layer)
        ttnn.deallocate(key_layer)

        attention_scores = ttnn.to_memory_config(attention_scores, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        scale_value = self.attention_head_size**-0.5
        attention_scores = ttnn.multiply(attention_scores, scale_value)
        attention_probs = ttnn.softmax(attention_scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attention_scores)
        attention_probs = ttnn.to_layout(attention_probs, ttnn.TILE_LAYOUT)

        value_layer = ttnn.to_memory_config(value_layer, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        attention_probs = ttnn.to_memory_config(
            attention_probs,
            memory_config=ttnn.create_sharded_memory_config(
                attention_probs.shape,
                core_grid=ttnn.CoreGrid(y=mm_d_y, x=mm_d_x),
                strategy=mm_d_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        context_layer = ttnn.matmul(
            attention_probs,
            value_layer,
            memory_config=mm_d_x_memory_config,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(value)
        ttnn.deallocate(value_layer)

        if not output_attentions:
            ttnn.deallocate(attention_probs)
        else:
            attention_probs = ttnn.to_memory_config(attention_probs, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        if self.num_attention_heads > 1:
            context_layer = ttnn.to_memory_config(context_layer, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            context_layer = ttnn.experimental.nlp_concat_heads(
                context_layer, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
            )

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
