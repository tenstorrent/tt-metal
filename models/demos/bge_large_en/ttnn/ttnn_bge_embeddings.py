# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import is_blackhole

if is_blackhole():
    pass
else:
    pass


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
        # Encoder layers require sharded inputs, so we must always shard
        # For small batches, we need to ensure dimensions are compatible
        *batch_sizes, height, width = embeddings.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        shard_height = batch_size * height
        shard_width = width
        core_grid = ttnn.CoreGrid(y=8, x=8)

        # Always try to shard - encoder layers require it
        # If sharding fails, it means the batch/seq_len combination is incompatible
        try:
            # Check if dimensions are divisible by core grid for BLOCK sharding
            if shard_height % core_grid.y == 0 and shard_width % core_grid.x == 0:
                embeddings = ttnn.to_memory_config(
                    embeddings,
                    memory_config=ttnn.create_sharded_memory_config(
                        embeddings.shape,
                        core_grid=core_grid,
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat8_b,
                )
            else:
                # Dimensions don't align - this will cause issues downstream
                # For vLLM, we should ensure batch sizes are compatible
                # Try anyway - might work with internal padding
                embeddings = ttnn.to_memory_config(
                    embeddings,
                    memory_config=ttnn.create_sharded_memory_config(
                        embeddings.shape,
                        core_grid=core_grid,
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat8_b,
                )
        except RuntimeError as e:
            # If sharding fails, this is a fundamental incompatibility
            # For vLLM, batch sizes should be chosen to avoid this
            raise RuntimeError(
                f"Cannot shard embeddings tensor: shape={embeddings.shape}, "
                f"shard_height={shard_height}, shard_width={shard_width}, "
                f"core_grid={core_grid}. Error: {e}"
            ) from e

        # Calculate LayerNorm program config dynamically based on tensor dimensions
        # block_h must equal M (in tiles) / num_cores_r
        # For sharded tensors: M = total_height (batch_size * seq_len), num_cores_r = core_grid.y
        # After unsqueeze, embeddings shape is [batch_size, 1, seq_len, hidden_size]
        # For block sharding, we need to calculate based on the sharded dimensions
        *batch_sizes, height, width = embeddings.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs

        # Calculate M in tiles: total height (batch_size * height) divided by tile height (32)
        # Then divide by number of cores in Y direction (8)
        M_tiles = (batch_size * height) // 32  # Convert to tiles (tile height = 32)
        core_grid_y = 8  # From layernorm_program_config
        block_h = M_tiles // core_grid_y

        # Ensure block_h is valid (must be > 0)
        if block_h == 0:
            block_h = 1

        # Create dynamic program config
        dynamic_layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=4,  # Keep same as original (1024 / 32 / 8 = 4)
            block_h=block_h,  # Calculate dynamically: M_tiles / core_grid_y
            block_w=4,  # 1024 / 32 / 8 = 4 (keep same)
            inplace=True,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )

        # Use sharded memory config and program config (always sharded now)
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
