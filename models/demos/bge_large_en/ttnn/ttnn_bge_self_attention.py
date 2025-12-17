# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.bge_large_en.ttnn.common import TILE_HEIGHT, core_grid_8x8, dim_t__x, head_size_t


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
        *batch_sizes, height, width = hidden_states.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        head_size = width // num_heads  # head_size in elements

        # Calculate sequence length and related values
        seq_len = height
        seqL_padded = (((seq_len - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT
        seqL_t = seqL_padded // TILE_HEIGHT

        if seq_len <= 384:
            seqL_factor = 1
        else:
            seqL_factor = 2

        # Calculate per_core_M for query_key_value matmul
        M_tiles = (batch_size * seq_len) // TILE_HEIGHT
        per_core_M_qkv = M_tiles // core_grid_8x8.y
        if per_core_M_qkv == 0:
            per_core_M_qkv = 1

        # Create dynamic query_key_value program config
        dynamic_query_key_value_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=dim_t__x // seqL_factor,  # 4 for seq_len <= 384, 2 for seq_len > 384
            out_subblock_h=1,
            out_subblock_w=dim_t__x // seqL_factor,  # 4 for seq_len <= 384, 2 for seq_len > 384
            per_core_M=per_core_M_qkv,  # Calculate dynamically
            per_core_N=dim_t__x * 3,  # Keep same (1024 / 32 / 8 * 3 = 12)
            transpose_mcast=False,
            fused_activation=None,
        )

        head_seqL_t__x = (num_heads * seqL_t) // core_grid_8x8.x

        query_key_value_output = ttnn.linear(
            hidden_states,
            self.parameters.query_key_value.weight,
            bias=self.parameters.query_key_value.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=dynamic_query_key_value_matmul_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,  # FP32 accumulation per engineering guidance (attention weights)
            ),
            dtype=ttnn.bfloat8_b,  # Keep BF8 for storage; compute will use BF16 with FP32 accumulation
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
        value_layer = ttnn.reallocate(value_layer)
        query_layer = ttnn.reallocate(query_layer)
        key_layer = ttnn.reallocate(key_layer)

        # Create dynamic pre_softmax config
        dynamic_pre_softmax_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=head_size_t,  # head_size in tiles
            out_subblock_h=1,
            out_subblock_w=seqL_t // 2,  # Dynamic based on seq_len
            per_core_M=head_seqL_t__x,  # Calculate dynamically
            per_core_N=seqL_t,  # Calculate dynamically
        )

        attention_scores = ttnn.matmul(
            query_layer,
            key_layer,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,  # BF16 compute per engineering guidance (attention scores)
            program_config=dynamic_pre_softmax_config,
        )
        ttnn.deallocate(query_layer)
        ttnn.deallocate(key_layer)
        attention_scores = ttnn.reallocate(attention_scores)

        # Create dynamic softmax config
        dynamic_softmax_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            subblock_w=seqL_t,  # Dynamic based on seq_len
            block_h=head_seqL_t__x,  # Calculate dynamically
            block_w=seqL_t,  # Calculate dynamically
        )

        attention_probabilities = ttnn.transformer.attention_softmax_(
            attention_scores,
            attention_mask=attention_mask,
            head_size=head_size,  # head_size in elements
            program_config=dynamic_softmax_config,
        )
        context_layer = ttnn.matmul(
            attention_probabilities,
            value_layer,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,  # BF16 compute per engineering guidance (attention output)
        )

        ttnn.deallocate(attention_probabilities)
        ttnn.deallocate(value_layer)
        context_layer = ttnn.reallocate(context_layer)

        context_layer = ttnn.experimental.nlp_concat_heads(
            context_layer,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        )
        return context_layer
