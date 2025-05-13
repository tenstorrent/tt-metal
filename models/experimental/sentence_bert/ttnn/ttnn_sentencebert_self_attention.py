# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.common import query_key_value_matmul_program_config


class TtnnSentenceBertSelfAttention:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.query = ttnn.linear
        self.key = ttnn.linear
        self.value = ttnn.linear
        self.num_attention_heads = config.num_attention_heads

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        query_key_value_output = ttnn.linear(
            hidden_states,
            self.parameters.query_key_value.weight,
            bias=self.parameters.query_key_value.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=query_key_value_matmul_program_config,
        )
        query_key_value_output_dram = ttnn.to_memory_config(query_key_value_output, ttnn.DRAM_MEMORY_CONFIG)
        (
            query_layer,
            key_layer,
            value_layer,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            query_key_value_output_dram,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.config.num_attention_heads,
        )
        ttnn.deallocate(query_key_value_output)
        ttnn.deallocate(query_key_value_output_dram)
        key_layer = ttnn.permute(key_layer, (0, 1, 3, 2))
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            is_causal=False,
        )
        ttnn.deallocate(query_layer)
        ttnn.deallocate(key_layer)
        ttnn.deallocate(value_layer)

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(
            attn_output, (attn_output.shape[0], attn_output.shape[1], attn_output.shape[2] * attn_output.shape[3])
        )
        return attn_output
