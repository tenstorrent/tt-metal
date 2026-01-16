# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.bge_large_en.ttnn.common import TILE_HEIGHT, core_grid_8x8, dim_t__x


class TtnnBGEEmbeddings:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.word_embeddings = ttnn.embedding
        self.position_embeddings = ttnn.embedding
        self.token_type_embeddings = ttnn.embedding
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, input_ids: ttnn.Tensor, token_type_ids: ttnn.Tensor, position_ids: ttnn.Tensor, device):
        if input_ids.is_sharded():
            input_ids_interleaved = ttnn.sharded_to_interleaved(input_ids, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(input_ids)
        else:
            input_ids_interleaved = input_ids
        word_embeddings = self.word_embeddings(
            input_ids_interleaved,
            weight=self.parameters.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            padding_idx=self.config.pad_token_id,
        )

        if token_type_ids.is_sharded():
            token_type_ids_interleaved = ttnn.sharded_to_interleaved(token_type_ids, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(token_type_ids)
        else:
            token_type_ids_interleaved = token_type_ids

        token_type_embeddings = self.token_type_embeddings(
            token_type_ids_interleaved,
            self.parameters.token_type_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if position_ids.is_sharded():
            position_ids_interleaved = ttnn.sharded_to_interleaved(position_ids, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(position_ids)
        else:
            position_ids_interleaved = position_ids

        position_embeddings = self.position_embeddings(
            position_ids_interleaved,
            self.parameters.position_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        ttnn.deallocate(word_embeddings)
        ttnn.deallocate(token_type_embeddings)
        ttnn.deallocate(position_embeddings)
        embeddings = ttnn.unsqueeze(embeddings, dim=1)

        # BGE-large uses 8x8 grid (vs 6x8 for sentence_bert)
        # This is because 1024 (hidden_size) % 8 == 0, but 1024 % 6 != 0
        embeddings = ttnn.to_memory_config(
            embeddings,
            memory_config=ttnn.create_sharded_memory_config(
                embeddings.shape,
                core_grid=core_grid_8x8,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        # Calculate block_h dynamically based on actual tensor shape for different sequence lengths
        *batch_sizes, height, width = embeddings.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // TILE_HEIGHT
        block_h = M_tiles // core_grid_8x8.y
        if block_h == 0:
            block_h = 1
        # Create dynamic program config matching the static one but with correct block_h
        dynamic_layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            subblock_w=dim_t__x,
            block_h=block_h,
            block_w=dim_t__x,
            inplace=True,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )
        embeddings = self.LayerNorm(
            embeddings,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            program_config=dynamic_layernorm_program_config,
        )
        return embeddings
