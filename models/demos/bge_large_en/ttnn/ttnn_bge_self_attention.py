# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.common.utility_functions import is_blackhole

if is_blackhole():
    pass
else:
    pass


class TtnnBGESelfAttention:
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
        # Calculate per_core_M dynamically based on tensor dimensions
        *batch_sizes, height, width = hidden_states.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32  # Convert to tiles (tile height = 32)
        core_grid_y = 8
        per_core_M = M_tiles // core_grid_y
        if per_core_M == 0:
            per_core_M = 1

        # Determine if we need DRAM for QKV (for large sequences)
        # For seq_len >= 512, use DRAM to avoid L1 buffer clashes (like metal_BERT BFLOAT16-L1 config)
        total_height = batch_size * height
        use_dram_for_qkv = total_height >= 2048  # Threshold for large sequences (seq_len=512 with batch=8)

        # Ensure weights and bias are in DRAM (metal_BERT always uses DRAM for weights)
        qkv_weight = self.parameters.query_key_value.weight
        qkv_bias = self.parameters.query_key_value.bias

        if qkv_weight.memory_config().buffer_type != ttnn.BufferType.DRAM:
            qkv_weight_dram = ttnn.to_memory_config(qkv_weight, ttnn.DRAM_MEMORY_CONFIG)
        else:
            qkv_weight_dram = qkv_weight

        if qkv_bias.memory_config().buffer_type != ttnn.BufferType.DRAM:
            qkv_bias_dram = ttnn.to_memory_config(qkv_bias, ttnn.DRAM_MEMORY_CONFIG)
        else:
            qkv_bias_dram = qkv_bias

        # For large sequences, also convert input to DRAM to reduce L1 buffer allocations
        if use_dram_for_qkv:
            hidden_states_dram = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states_dram = hidden_states

        # Try reducing in0_block_w to reduce L1 buffer allocations
        # K_tiles = 1024 / 32 = 32, per core in X = 32 / 8 = 4
        # Options: 4, 2, 1. Try 2 to reduce L1 usage
        dynamic_query_key_value_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # Reduced from 4 to reduce L1 buffer allocations
            out_subblock_h=1,  # Must be 1 to satisfy out_subblock_w * out_subblock_h <= 4
            out_subblock_w=4,  # 1 * 4 = 4, which satisfies the constraint
            per_core_M=per_core_M,  # Calculate dynamically
            per_core_N=12,  # Match BGE: (3072 / 32 / 8 = 12)
            transpose_mcast=False,
            fused_activation=None,
        )

        # Always use DRAM for large sequences to avoid L1 buffer clashes
        qkv_output_mem_config = ttnn.DRAM_MEMORY_CONFIG if use_dram_for_qkv else ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

        query_key_value_output = ttnn.linear(
            hidden_states_dram,
            qkv_weight_dram,
            bias=qkv_bias_dram,
            memory_config=qkv_output_mem_config,
            program_config=dynamic_query_key_value_matmul_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,  # FP32 accumulation per engineering guidance (attention weights)
            ),
            dtype=ttnn.bfloat8_b,  # Keep BF8 for storage; compute will use BF16 with FP32 accumulation
        )

        # Deallocate DRAM input if we converted it
        if use_dram_for_qkv:
            hidden_states_dram.deallocate()

        # Convert QKV output from DRAM to sharded if needed for split_heads
        # split_query_key_value_and_split_heads expects sharded input, so create proper sharded config
        if use_dram_for_qkv:
            # Create proper sharded memory config for QKV output
            qkv_sharded_config = ttnn.create_sharded_memory_config(
                query_key_value_output.shape,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            query_key_value_output = ttnn.to_memory_config(query_key_value_output, qkv_sharded_config)
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

        # Get sequence length to determine if we should use composite SDPA
        *batch_sizes, attn_seq_len, head_dim = query_layer.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs

        # For seq_len >= 512, follow metal_BERT_large_11 pattern:
        # Use separate matmuls with DRAM memory config instead of composite SDPA
        # This avoids L1 OOM issues and mask shape complications
        use_dram_for_large_seq = attn_seq_len >= 512

        if use_dram_for_large_seq:
            # Convert Q, K, V to DRAM for large sequence length
            query_layer = ttnn.to_memory_config(query_layer, ttnn.DRAM_MEMORY_CONFIG)
            key_layer = ttnn.to_memory_config(key_layer, ttnn.DRAM_MEMORY_CONFIG)
            value_layer = ttnn.to_memory_config(value_layer, ttnn.DRAM_MEMORY_CONFIG)

            # Calculate per_core_M and per_core_N dynamically for attention scores
            # Q @ K^T: [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, seq_len]
            # Output: [batch, heads, seq_len, seq_len] -> HEIGHT_SHARDED
            # For HEIGHT_SHARDED output, per_core_M must match the shard height in tiles
            query_shape = query_layer.shape
            batch_size = query_shape[0]
            num_heads = query_shape[1]
            seq_len = query_shape[2]

            M_total = batch_size * num_heads * seq_len  # Total M elements = 8 * 16 * 512 = 65536
            N_total = seq_len  # Total N elements = 512

            core_grid_y = 8
            core_grid_x = 8
            total_cores = core_grid_x * core_grid_y  # 64 cores

            # For HEIGHT_SHARDED output, the M dimension is sharded across all cores
            # shard_height = M_total / total_cores = 65536 / 64 = 1024 elements
            # per_core_M = shard_height / tile_height = 1024 / 32 = 32 tiles
            shard_height_elements = M_total // total_cores  # 1024 elements per core
            per_core_M_attn = shard_height_elements // 32  # 32 tiles per core
            if per_core_M_attn == 0:
                per_core_M_attn = 1

            # For MatmulMultiCoreReuseProgramConfig, per_core_N is the total N in tiles
            per_core_N_attn = N_total // 32  # 512 / 32 = 16 tiles

            # Choose out_subblock_w that divides per_core_N_attn evenly
            # For seq_len=512: per_core_N = 16, valid divisors: 1, 2, 4, 8, 16
            # For seq_len=384: per_core_N = 12, valid divisors: 1, 2, 3, 4, 6, 12
            if per_core_N_attn % 8 == 0:
                out_subblock_w_attn = 8
            elif per_core_N_attn % 4 == 0:
                out_subblock_w_attn = 4
            elif per_core_N_attn % 2 == 0:
                out_subblock_w_attn = 2
            else:
                out_subblock_w_attn = 1

            # Create dynamic program config for attention scores matmul
            dynamic_pre_softmax_config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(core_grid_x, core_grid_y),
                in0_block_w=2,  # head_dim / 32 / 4 = 64 / 32 / 1 = 2
                out_subblock_h=1,
                out_subblock_w=out_subblock_w_attn,
                per_core_M=per_core_M_attn,
                per_core_N=per_core_N_attn,
            )

            # Matmul: Q @ K^T -> attention_scores
            # Use BFLOAT8_B dtype and HEIGHT_SHARDED memory config as specified (OP3_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG)
            attention_scores = ttnn.matmul(
                query_layer,
                key_layer,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,  # OP3_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG: HEIGHT_SHARDED, L1
                dtype=ttnn.bfloat8_b,  # OP3_PRE_SOFTMAX_BMM_OUTPUT_DTYPE: BFLOAT8_B
                program_config=dynamic_pre_softmax_config,
            )
            ttnn.deallocate(query_layer)
            ttnn.deallocate(key_layer)

            # attention_scores is already in L1_HEIGHT_SHARDED_MEMORY_CONFIG from matmul output
            # No conversion needed - it's already in the correct memory config for softmax

            # Use scale_mask_softmax_in_place exactly like metal_BERT_large_11
            # Following metal_BERT pattern exactly: reshape using padded_shape, then softmax, then reshape back
            shape = attention_scores.padded_shape  # Use padded_shape like metal_BERT
            attn_scores_shape = attention_scores.shape  # Store shape for later use
            attention_scores_reshaped = attention_scores.reshape(shape[0], 1, shape[1] * shape[2], shape[3])

            # Calculate scale factor exactly like metal_BERT: 1 / sqrt(head_size)
            freciprocal_of_sqrt_hidden_dim = 1.0 / math.sqrt(head_size)

            # Calculate block_h and block_w for reshaped tensor
            # Reshaped shape: [batch, 1, heads*seq_len, seq_len]
            # M = batch * heads * seq_len (total height after reshape)
            # K = seq_len (width dimension)
            # For sharded softmax: block_h = (M / num_cores) / tile_height, block_w = K / tile_width
            batch_size = shape[0]
            num_heads = shape[1]
            seq_len = shape[2]  # Same as shape[3]
            total_cores_softmax = core_grid_x * core_grid_y  # 64 cores (8x8)

            M_total = batch_size * num_heads * seq_len  # Total M after reshape
            M_per_core = M_total // total_cores_softmax  # M per core
            block_h_softmax = M_per_core // 32  # Convert to tiles (tile height = 32)
            if block_h_softmax == 0:
                block_h_softmax = 1

            # block_w = seq_len in tiles
            block_w_softmax = seq_len // 32
            if block_w_softmax == 0:
                block_w_softmax = 1

            # Choose subblock_w that divides block_w_softmax (like metal_BERT uses subblock_w=6 for block_w=12)
            # For block_w=16, use subblock_w that divides evenly
            if block_w_softmax % 8 == 0:
                subblock_w_softmax = 8
            elif block_w_softmax % 4 == 0:
                subblock_w_softmax = 4
            elif block_w_softmax % 2 == 0:
                subblock_w_softmax = 2
            else:
                subblock_w_softmax = 1

            # Create softmax config exactly like metal_BERT sharded config pattern
            softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_x, core_grid_y),
                subblock_w=subblock_w_softmax,
                block_h=block_h_softmax,
                block_w=block_w_softmax,
            )

            # Apply scale_mask_softmax_in_place exactly like metal_BERT
            # API: scale_mask_softmax_in_place(input, scale, mask, program_config)
            attention_probabilities_reshaped = ttnn.scale_mask_softmax_in_place(
                attention_scores_reshaped,
                freciprocal_of_sqrt_hidden_dim,
                attention_mask,
                program_config=softmax_program_config,
            )

            # Reshape back to original shape exactly like metal_BERT
            attention_probabilities = attention_probabilities_reshaped.reshape(shape)

            # Matmul: attention_probabilities @ V -> context_layer
            # Following metal_BERT_large_11 pattern: use MatmulMultiCoreReuseProgramConfig (NOT MultiCast)
            # attention_probabilities: [8, 16, 512, 512] (sharded)
            # value_layer: [8, 16, 512, 64] (in DRAM)
            # output: [8, 16, 512, 64] (sharded)

            # CRITICAL: per_core_M must match the actual shard height of attention_probabilities!
            # attention_probabilities is sharded with shard_height = (batch*heads*seq_len) / total_cores
            # For [8, 16, 512, 512] with 8x8=64 cores:
            #   shard_height = (8 * 16 * 512) / 64 = 1024 elements = 32 tiles
            #   So per_core_M MUST be 32 (shard height in tiles), NOT 256!

            # Get the actual shard shape from attention_probabilities
            # The shard height is: (batch * heads * seq_len) / (core_grid_x * core_grid_y)
            total_height = attn_scores_shape[0] * attn_scores_shape[1] * attn_scores_shape[2]
            total_cores = core_grid_x * core_grid_y
            shard_height_elements = total_height // total_cores  # 65536 / 64 = 1024
            per_core_M_post = shard_height_elements // 32  # 1024 / 32 = 32 tiles

            # K dimension: seq_len = 512, in tiles = 512/32 = 16
            K_tiles_total = attn_scores_shape[3] // 32  # 512/32 = 16

            # N dimension: head_dim = 64, in tiles = 64/32 = 2
            N_tiles_total = head_dim // 32  # 64/32 = 2

            # per_core_N: for small N (2 tiles), typically 2 per core
            per_core_N_post = N_tiles_total  # 2 tiles per core

            # Choose out_subblock_w that divides per_core_N_post
            if per_core_N_post % 2 == 0:
                out_subblock_w_post = 2
            else:
                out_subblock_w_post = 1

            # Choose out_subblock_h - typically 4 for attention (as per metal_BERT)
            out_subblock_h_post = 4

            # Create dynamic program config for post-softmax matmul
            # Following metal_BERT_large_11: MatmulMultiCoreReuseProgramConfig (NOT MultiCast)
            dynamic_post_softmax_config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(core_grid_x, core_grid_y),
                in0_block_w=K_tiles_total,  # TOTAL K tiles (16), NOT per-core!
                out_subblock_h=out_subblock_h_post,
                out_subblock_w=out_subblock_w_post,
                per_core_M=per_core_M_post,
                per_core_N=per_core_N_post,
            )

            # Matmul: attention_probabilities @ V -> context_layer
            # Use BFLOAT8_B dtype and HEIGHT_SHARDED memory config as specified (OP5_POST_SOFTMAX_BMM_OUTPUT_MEMCFG)
            context_layer = ttnn.matmul(
                attention_probabilities,
                value_layer,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,  # OP5_POST_SOFTMAX_BMM_OUTPUT_MEMCFG: HEIGHT_SHARDED, L1
                dtype=ttnn.bfloat8_b,  # OP5_POST_SOFTMAX_BMM_OUTPUT_DTYPE: BFLOAT8_B
                program_config=dynamic_post_softmax_config,
            )

            ttnn.deallocate(attention_probabilities)
            ttnn.deallocate(value_layer)

            # context_layer is already in L1_HEIGHT_SHARDED_MEMORY_CONFIG from matmul output
            # No conversion needed - it's already in the correct memory config for concat_heads
        else:
            # Original path for seq_len < 512: use separate matmuls with L1 memory
            # Calculate per_core_M and per_core_N dynamically for attention scores
            M_tiles = attn_seq_len // 32  # Total tiles in M dimension
            N_tiles = attn_seq_len // 32  # Total tiles in N dimension

            # Original config pattern for seq_len=384: per_core_M=24 (2x M_tiles=12), per_core_N=12 (equals N_tiles)
            if M_tiles <= 16:
                per_core_M_attn = M_tiles * 2
                if per_core_M_attn % M_tiles != 0:
                    per_core_M_attn = M_tiles
            else:
                per_core_M_attn = max(1, M_tiles // 8)

            per_core_N_attn = N_tiles

            # Create dynamic program config for attention scores
            dynamic_pre_softmax_config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=6,
                per_core_M=per_core_M_attn,
                per_core_N=per_core_N_attn,
            )

            # Use BFLOAT8_B dtype as specified in config (OP3_PRE_SOFTMAX_BMM_OUTPUT_DTYPE)
            attention_scores = ttnn.matmul(
                query_layer,
                key_layer,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,  # OP3_PRE_SOFTMAX_BMM_OUTPUT_DTYPE: BFLOAT8_B
                program_config=dynamic_pre_softmax_config,
            )
            ttnn.deallocate(query_layer)
            ttnn.deallocate(key_layer)

            # Calculate block_h dynamically for softmax
            *batch_sizes, softmax_seq_len, _ = attention_scores.shape
            batch_size = 1
            for bs in batch_sizes:
                batch_size *= bs
            M_tiles_softmax = (batch_size * softmax_seq_len) // 32
            core_grid_y = 8
            block_h_softmax = M_tiles_softmax // core_grid_y
            if block_h_softmax == 0:
                block_h_softmax = 1

            block_w_softmax = softmax_seq_len // 32
            if block_w_softmax == 0:
                block_w_softmax = 1

            # Create dynamic softmax config
            dynamic_softmax_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                subblock_w=6,
                block_h=block_h_softmax,
                block_w=block_w_softmax,
            )

            # Use scale_mask_softmax_in_place exactly like metal_BERT_large_11
            # Following metal_BERT pattern exactly: reshape using padded_shape, then softmax, then reshape back
            shape = attention_scores.padded_shape  # Use padded_shape like metal_BERT
            attention_scores_reshaped = attention_scores.reshape(shape[0], 1, shape[1] * shape[2], shape[3])

            # Calculate scale factor exactly like metal_BERT: 1 / sqrt(head_size)
            freciprocal_of_sqrt_hidden_dim = 1.0 / math.sqrt(head_size)

            # Apply scale_mask_softmax_in_place exactly like metal_BERT
            # API: scale_mask_softmax_in_place(input, scale, mask, program_config)
            attention_probabilities_reshaped = ttnn.scale_mask_softmax_in_place(
                attention_scores_reshaped,
                freciprocal_of_sqrt_hidden_dim,
                attention_mask,
                program_config=dynamic_softmax_config,
            )

            # Reshape back to original shape exactly like metal_BERT
            attention_probabilities = attention_probabilities_reshaped.reshape(shape)
            # Use BFLOAT8_B dtype as specified in config (OP5_POST_SOFTMAX_BMM_OUTPUT_DTYPE)
            context_layer = ttnn.matmul(
                attention_probabilities,
                value_layer,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,  # OP5_POST_SOFTMAX_BMM_OUTPUT_DTYPE: BFLOAT8_B
            )
            ttnn.deallocate(attention_probabilities)
            ttnn.deallocate(value_layer)
        context_layer = ttnn.experimental.nlp_concat_heads(
            context_layer,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        )

        # Convert to ROW_MAJOR orientation if needed (nlp_concat_heads may output COL_MAJOR)
        # Self-output linear requires ROW_MAJOR for non-transpose MCAST
        if context_layer.is_sharded():
            mem_config = context_layer.memory_config()
            if (
                mem_config.shard_spec is not None
                and mem_config.shard_spec.orientation == ttnn.ShardOrientation.COL_MAJOR
            ):
                # Create a new memory config with ROW_MAJOR orientation
                shard_spec = mem_config.shard_spec
                shard_height = shard_spec.shape[0]
                shard_width = shard_spec.shape[1]

                row_major_config = ttnn.create_sharded_memory_config(
                    shape=(shard_height, shard_width),
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                context_layer = ttnn.to_memory_config(context_layer, row_major_config)

        return context_layer
