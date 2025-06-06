# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.common import (
    query_key_value_matmul_program_config,
    pre_softmax_config,
    softmax_config,
)


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
        num_heads = self.config.num_attention_heads
        *_, hidden_size = hidden_states.shape
        head_size = hidden_size // num_heads
        query_key_value_output = ttnn.linear(
            hidden_states,
            self.parameters.query_key_value.weight,
            bias=self.parameters.query_key_value.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=query_key_value_matmul_program_config,
            dtype=ttnn.bfloat8_b,
        )
        (
            query_layer,
            key_layer,
            value_layer,
        ) = ttnn.experimental.split_query_key_value_and_split_heads(
            query_key_value_output,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            num_heads=num_heads,
        )
        ttnn.deallocate(query_key_value_output)
        attention_scores = ttnn.matmul(
            query_layer,
            key_layer,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=pre_softmax_config,
        )
        ttnn.deallocate(query_layer)
        ttnn.deallocate(key_layer)
        attention_probabilities = ttnn.transformer.attention_softmax_(
            attention_scores,
            attention_mask=attention_mask,
            head_size=head_size,
            program_config=softmax_config,
        )
        context_layer = ttnn.matmul(
            attention_probabilities,
            value_layer,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(attention_probabilities)
        ttnn.deallocate(value_layer)
        context_layer = ttnn.experimental.nlp_concat_heads(
            context_layer,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        )
        return context_layer
