# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.experimental.functional_segformer.tt.common import Conv


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

    def transpose_for_scores(self, hidden_states):
        # new_shape = tuple(hidden_states.shape)[:-1] + (self.num_attention_heads, self.attention_head_size)
        new_shape = (
            hidden_states.shape[0],
            hidden_states.shape[-2],
            self.num_attention_heads,
            self.attention_head_size,
        )
        device = hidden_states.device()
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, new_shape)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device)

        if len(hidden_states.shape) == 5:
            output = ttnn.permute(hidden_states, (0, 1, 3, 2, 4))
        elif len(hidden_states.shape) == 4:
            output = ttnn.permute(hidden_states, (0, 2, 1, 3))
        if len(hidden_states.shape) == 3:
            output = ttnn.permute(hidden_states, (0, 2, 1))
        ttnn.deallocate(hidden_states)

        return output

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        height: int,
        width: int,
        parameters,
        output_attentions=False,
    ):
        device = hidden_states.device()

        mm_a_x_strategy = ttnn.ShardStrategy.HEIGHT
        mm_a_x_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        mm_d_x_strategy = mm_a_x_strategy
        mm_d_x_memory_config = mm_a_x_memory_config
        mm_a_y = 8
        if (hidden_states.shape[-2] == 256) and (hidden_states.shape[-1] == 256):
            mm_a_x = 8
            mm_b_x = 8
            mm_d_x = 2
            mm_d_y = 4
            mm_e_x = 8
            mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (hidden_states.shape[-2] == 1024) and (hidden_states.shape[-1] == 160):
            mm_a_x = 5
            mm_b_x = 5
            mm_d_x = 5
            mm_d_y = 4
            mm_e_x = 8
            mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif (hidden_states.shape[-2] == 4096) and (hidden_states.shape[-1] == 64):
            mm_a_x = 8  # 8
            mm_b_x = 1  # 1
            mm_d_x = 4
            mm_d_y = 8
            mm_e_x = 8
        elif (hidden_states.shape[-2] == 16384) and (hidden_states.shape[-1] == 32):
            mm_a_x = 8  # 8
            mm_b_x = 1  # 1
            mm_d_x = 4
            mm_d_y = 8
            mm_e_x = 8

        # print("mm-1--", hidden_states.shape, parameters.query.weight.shape)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                strategy=mm_a_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        query = ttnn.linear(
            hidden_states,
            parameters.query.weight,
            bias=parameters.query.bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            dtype=ttnn.bfloat8_b,
        )

        # print("Q1", query.shape)
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        # Split Heads
        if self.num_attention_heads == 1:
            query_layer = query
        else:
            query_layer = self.transpose_for_scores(query)

        # print("Q2", query_layer.shape)

        # print("sr0", hidden_states.shape)
        if self.sr_ratio > 1:
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, num_channels = hidden_states.shape
            elif len(hidden_states.shape) == 4:
                batch_size, __, seq_len, num_channels = hidden_states.shape

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, num_channels))

            # print("sr1", hidden_states.shape)
            hidden_states, __, __ = self.sr(device, hidden_states)
            # print("sr2", hidden_states.shape)
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

            # print("sr3", hidden_states.shape)
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm.weight,
                bias=parameters.layer_norm.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # print("mm-2--", hidden_states.shape, parameters.key.weight.shape)

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
        # hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # print("K1", key.shape)
        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        if self.num_attention_heads == 1:
            key_layer = key
        else:
            key_layer = self.transpose_for_scores(key)
        key_layer = ttnn.permute(key_layer, (0, 1, 3, 2))
        # print("K2", key_layer.shape)

        # print("mm-3--", hidden_states.shape, parameters.value.weight.shape)
        value = ttnn.linear(
            hidden_states,
            parameters.value.weight,
            bias=parameters.value.bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_b_x),
            dtype=ttnn.bfloat8_b,
        )
        # hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(hidden_states)
        # print("V1", value.shape)
        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        if self.num_attention_heads == 1:
            value_layer = value
        else:
            value_layer = self.transpose_for_scores(value)
        # print("V2", value_layer.shape)

        # print("mm-4--", query_layer.shape, key_layer.shape)

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
        attention_scores = ttnn.to_memory_config(attention_scores, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        denominator_value = ttnn.ones(
            attention_scores.shape, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        denominator_value = denominator_value * math.sqrt(self.attention_head_size)
        denominator_value = ttnn.reciprocal(denominator_value)
        attention_scores = attention_scores * denominator_value

        # Normalize the attention scores to probabilities.
        attention_probs = ttnn.softmax(attention_scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # print("mm-5--", attention_probs.shape, value_layer.shape)

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
            # core_grid=ttnn.CoreGrid(y=8, x=8),
            # program_config=ATTN_SCORE_MM_PROGCFG,
        )
        ttnn.deallocate(value)
        ttnn.deallocate(value_layer)

        if not output_attentions:
            ttnn.deallocate(attention_probs)
        else:
            attention_probs = ttnn.to_memory_config(attention_probs, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        # context_layer = ttnn.to_memory_config(context_layer, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        context_layer = ttnn.to_memory_config(context_layer, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        # print("cxt1", context_layer.shape)

        if self.num_attention_heads > 1:
            context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
            context_layer = ttnn.to_memory_config(
                context_layer, ttnn.L1_MEMORY_CONFIG
            )  # This throws OOM issue while runnning whole_model so, DRAM memory config is used.
            new_context_layer_shape = tuple(context_layer.shape)[:-2] + (self.all_head_size,)
            context_layer = ttnn.from_device(context_layer)
            context_layer = ttnn.to_layout(context_layer, layout=ttnn.ROW_MAJOR_LAYOUT)
            context_layer = ttnn.reshape(context_layer, new_context_layer_shape)
            context_layer = ttnn.to_device(context_layer, device)
            context_layer = ttnn.to_layout(context_layer, layout=ttnn.TILE_LAYOUT)
        # print("cxt2", context_layer.shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
