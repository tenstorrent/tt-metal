# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnSentenceBertEmbeddings:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.word_embeddings = ttnn.embedding
        self.position_embeddings = ttnn.embedding
        self.token_type_embeddings = ttnn.embedding
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, input_ids: ttnn.Tensor, token_type_ids: ttnn.Tensor, position_ids: ttnn.Tensor, device):
        if input_ids.is_sharded():
            input_ids = ttnn.sharded_to_interleaved(input_ids, ttnn.L1_MEMORY_CONFIG)

        word_embeddings = self.word_embeddings(
            input_ids,
            weight=self.parameters.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            padding_idx=self.config.pad_token_id,
        )

        token_type_embeddings = self.token_type_embeddings(
            token_type_ids,
            self.parameters.token_type_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        embeddings = word_embeddings + token_type_embeddings

        position_embeddings = self.position_embeddings(
            position_ids,
            self.parameters.position_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        ttnn.deallocate(word_embeddings)
        ttnn.deallocate(token_type_embeddings)
        ttnn.deallocate(position_embeddings)

        embeddings = self.LayerNorm(
            embeddings,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        )
        return embeddings
