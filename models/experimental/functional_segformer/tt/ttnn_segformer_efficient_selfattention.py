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
        new_shape = tuple(hidden_states.shape)[:-1] + (self.num_attention_heads, self.attention_head_size)
        device = hidden_states.device()
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, new_shape)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device)

        if len(hidden_states.shape) == 4:
            output = ttnn.permute(hidden_states, (0, 2, 1, 3))
        else:
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
        # query = ttnn.linear(
        #     hidden_states,
        #     parameters.query.weight,
        #     bias=parameters.query.bias,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     core_grid=ttnn.CoreGrid(y=8, x=12),
        # )
        print("mm-1--", hidden_states.shape)
        if hidden_states.shape[1] >= 2048:
            mm_1_y = 8
            mm_1_x = 4
        elif hidden_states.shape[1] == 1024:
            mm_1_y = 8
            mm_1_x = 5
        elif hidden_states.shape[1] == 256:
            mm_1_y = 8
            mm_1_x = 4

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        if hidden_states.shape[1] >= 2048:
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

            query = ttnn.linear(
                hidden_states,
                parameters.query.weight,
                bias=parameters.query.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                dtype=ttnn.bfloat8_b,
            )
        elif hidden_states.shape[1] == 1024:
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

            query = ttnn.linear(
                hidden_states,
                parameters.query.weight,
                bias=parameters.query.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                dtype=ttnn.bfloat8_b,
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

            query = ttnn.linear(
                hidden_states,
                parameters.query.weight,
                bias=parameters.query.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_1_y, x=mm_1_x),
                dtype=ttnn.bfloat8_b,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        query_layer = self.transpose_for_scores(query)

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width)
            hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
            hidden_states = ttnn.from_device(hidden_states)
            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch_size, num_channels, height, width))
            # Apply sequence reduction
            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
            hidden_states = self.sr(device, hidden_states)
            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.from_device(hidden_states)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = ttnn.reshape(hidden_states, (batch_size, num_channels, -1))

            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            if batch_size == 1:
                hidden_states = ttnn.reshape(
                    hidden_states, (batch_size, hidden_states.shape[0], hidden_states.shape[1])
                )
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm.weight,
                bias=parameters.layer_norm.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        print("mm-2--", hidden_states.shape)
        mm_2_y = 8
        # mm_2_x = 1
        mm_2_x_strategy = ttnn.ShardStrategy.BLOCK
        mm_2_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if hidden_states.shape[2] == 256:
            mm_2_x = 4
        else:
            mm_2_x = int(hidden_states.shape[2] / 32)
            if hidden_states.shape[2] == 32:
                mm_2_x_strategy = ttnn.ShardStrategy.HEIGHT
                mm_2_x_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=ttnn.CoreGrid(y=mm_2_y, x=mm_2_x),
                strategy=mm_2_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        # key = ttnn.linear(hidden_states, parameters.key.weight, bias=parameters.key.bias)
        key = ttnn.linear(
            hidden_states,
            parameters.key.weight,
            bias=parameters.key.bias,
            memory_config=mm_2_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_2_y, x=mm_2_x),
            dtype=ttnn.bfloat8_b,
        )
        # hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        key_layer = self.transpose_for_scores(key)

        print("mm-3--", hidden_states.shape)
        # value = ttnn.linear(
        #     hidden_states,
        #     parameters.value.weight,
        #     bias=parameters.value.bias,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        # )
        value = ttnn.linear(
            hidden_states,
            parameters.value.weight,
            bias=parameters.value.bias,
            memory_config=mm_2_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_2_y, x=mm_2_x),
            dtype=ttnn.bfloat8_b,
        )
        # hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(hidden_states)
        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        value_layer = self.transpose_for_scores(value)

        key_layer = ttnn.permute(key_layer, (0, 1, 3, 2))

        # attention_scores = ttnn.matmul(query_layer, key_layer)
        print("mm-4--", query_layer.shape, key_layer.shape)
        # ATTN_SCORE_MM_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=(8,8),
        #     in0_block_w=8,
        #     out_subblock_h=1,
        #     out_subblock_w=8,
        #     per_core_M=8,
        #     per_core_N=8,
        #     fuse_batch=True,
        #     fused_activation=None,
        #     mcast_in0=False,
        # )

        key_layer = ttnn.to_memory_config(key_layer, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        query_layer = ttnn.to_layout(query_layer, ttnn.TILE_LAYOUT)

        if query_layer.shape[1] < 5:
            mm_4_y = mm_1_y
            mm_4_x = mm_1_x
        elif query_layer.shape[1] == 5:
            mm_4_y = 4
            mm_4_x = 5
        elif query_layer.shape[1] > 5:
            mm_4_y = 4
            mm_4_x = 2

        query_layer = ttnn.to_memory_config(
            query_layer,
            memory_config=ttnn.create_sharded_memory_config(
                query_layer.shape,
                core_grid=ttnn.CoreGrid(y=mm_4_y, x=mm_4_x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        attention_scores = ttnn.matmul(
            query_layer,
            key_layer,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            # core_grid=ttnn.CoreGrid(y=mm_4_y, x=mm_4_x),
            # program_config=ATTN_SCORE_MM_PROGCFG,
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

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        print("mm-5--", attention_probs.shape, value_layer.shape)
        # context_layer = ttnn.matmul(
        #     attention_probs,
        #     value_layer,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     dtype=ttnn.bfloat16,
        #     core_grid=ttnn.CoreGrid(y=8, x=8),
        # )
        attention_probs = ttnn.to_layout(attention_probs, ttnn.TILE_LAYOUT)
        value_layer = ttnn.to_memory_config(value_layer, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        attention_probs = ttnn.to_memory_config(
            attention_probs,
            memory_config=ttnn.create_sharded_memory_config(
                attention_probs.shape,
                core_grid=ttnn.CoreGrid(y=mm_4_y, x=mm_4_x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        context_layer = ttnn.matmul(
            attention_probs,
            value_layer,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            # core_grid=ttnn.CoreGrid(y=8, x=8),
            # program_config=ATTN_SCORE_MM_PROGCFG,
        )

        ttnn.deallocate(value_layer)

        if not output_attentions:
            ttnn.deallocate(attention_probs)
        else:
            attention_probs = ttnn.to_memory_config(attention_probs, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        context_layer = ttnn.to_memory_config(context_layer, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
