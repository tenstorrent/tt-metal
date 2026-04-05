# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Optimized all-MiniLM-L6-v2 for Wormhole.

Key optimizations over the baseline (minilm_model.py):
  - Fused QKV projection (1 linear instead of 3)
  - L1 block-sharded memory for linear ops and layernorm
  - L1 height-sharded memory for attention score computation
  - bfloat8_b output dtype for linear ops (2x bandwidth reduction)
  - Wormhole compute kernel configs (HiFi2 for matmuls, HiFi4 for layernorm)
  - ttnn.experimental.split_query_key_value_and_split_heads / nlp_concat_heads

Architecture: 6 layers, hidden=384, heads=12, head_dim=32, intermediate=1536
Batch=8, seq=128 -> M=32 tiles, grid (4,8)
"""

import torch

import ttnn

# ---------------------------------------------------------------------------
# Program configs for MiniLM on Wormhole (batch=8, seq=128)
#
# Tile math:
#   M = (batch * seq) / 32 = (8 * 128) / 32 = 32 tiles
#   hidden_tiles = 384 / 32 = 12
#   intermediate_tiles = 1536 / 32 = 48
#   fused_qkv_tiles = 3 * 12 = 36
#
# Grid (6, 8), transpose_mcast=False:
#   per_core_M = M / 8 = 4
#   per_core_N depends on output width
# ---------------------------------------------------------------------------

GRID = (6, 8)

# QKV fused linear: [M, 12] x [12, 36] -> [M, 36]
qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID,
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=4,
    per_core_N=6,
    transpose_mcast=False,
    fused_activation=None,
)

# Self-output projection: [M, 12] x [12, 12] -> [M, 12]
self_out_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID,
    in0_block_w=4,
    out_subblock_h=2,
    out_subblock_w=2,
    per_core_M=4,
    per_core_N=2,
    transpose_mcast=False,
    fused_activation=None,
)

# FF1 (intermediate up-projection): [M, 12] x [12, 48] -> [M, 48]
ff1_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID,
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=8,
    per_core_M=4,
    per_core_N=8,
    transpose_mcast=False,
    fused_activation=(ttnn.UnaryOpType.GELU, True),
)

# FF2 (output down-projection): [M, 48] x [48, 12] -> [M, 12]
ff2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID,
    in0_block_w=4,
    out_subblock_h=2,
    out_subblock_w=2,
    per_core_M=4,
    per_core_N=2,
    transpose_mcast=False,
    fused_activation=None,
)

# LayerNorm: block_h = per_core_M = 4, block_w = hidden_tiles / grid_x
layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=GRID,
    subblock_w=2,
    block_h=4,
    block_w=2,
    inplace=True,
)

# Attention: Q*K^T pre-softmax matmul
# After head split: [batch*heads, seq, head_dim] = [96, 128, 32]
# In tiles: M_attn = (96*128)/32 = 384... no that's per-head.
# Actually split_query_key_value_and_split_heads produces height-sharded tensors
# with shape [batch, num_heads, seq, head_dim] = [8, 12, 128, 32]
# Attention scores: [8, 12, 128, 128] -> per head: [128, 128] in tiles = [4, 4]
# Total M for height shard: batch * heads * seq_tiles = 8 * 12 * 4 = 384 tiles
pre_softmax_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=GRID,
    in0_block_w=1,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=8,
    per_core_N=4,
)

softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=GRID,
    subblock_w=4,
    block_h=8,
    block_w=4,
)

# Compute kernel configs
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    packer_l1_acc=False,
)

COMPUTE_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
)


def custom_preprocessor(torch_model, name):
    """Fuse Q, K, V weights and biases into a single tensor for efficient projection."""
    parameters = {}
    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        qw = torch_model.query.weight
        kw = torch_model.key.weight
        vw = torch_model.value.weight
        qb = torch_model.query.bias
        kb = torch_model.key.bias
        vb = torch_model.value.bias

        # Transpose weights for ttnn linear
        qw = torch.transpose(qw, -1, -2)
        kw = torch.transpose(kw, -1, -2)
        vw = torch.transpose(vw, -1, -2)

        # Interleave heads: reshape to [hidden, num_heads//2, head_dim] then cat
        num_heads_half = 6  # 12 // 2
        const_w_dims = qw.shape[:-1]
        qw = qw.reshape([*const_w_dims, num_heads_half, -1])
        kw = kw.reshape(qw.shape)
        vw = vw.reshape(qw.shape)
        qkv_weight = torch.cat((qw, kw, vw), -1).reshape([*const_w_dims, -1])

        const_b_dims = qb.shape[:-1]
        qb = qb.reshape([*const_b_dims, num_heads_half, -1])
        kb = kb.reshape(qb.shape)
        vb = vb.reshape(qb.shape)
        qkv_bias = torch.cat((qb, kb, vb), -1).reshape([*const_b_dims, -1])

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = ttnn.from_torch(
            qkv_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        parameters["query_key_value"]["bias"] = ttnn.from_torch(qkv_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    return parameters


class MiniLMOptimized:
    """Optimized all-MiniLM-L6-v2 with sharded memory and fused QKV."""

    def __init__(self, config, parameters):
        self.config = config
        self.parameters = parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

    def embeddings(self, input_ids, token_type_ids, position_ids, device):
        """Word + position + token_type embeddings -> LayerNorm."""
        MEM = ttnn.DRAM_MEMORY_CONFIG

        word_emb = ttnn.embedding(
            input_ids,
            self.parameters.embeddings.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=MEM,
            padding_idx=self.config.pad_token_id,
        )
        token_type_emb = ttnn.embedding(
            token_type_ids,
            self.parameters.embeddings.token_type_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=MEM,
        )
        position_emb = ttnn.embedding(
            position_ids,
            self.parameters.embeddings.position_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=MEM,
        )

        # 4D for add
        word_emb = ttnn.unsqueeze(word_emb, dim=1)
        token_type_emb = ttnn.unsqueeze(token_type_emb, dim=1)
        position_emb = ttnn.unsqueeze(position_emb, dim=1)

        embeddings = ttnn.add(word_emb, token_type_emb, memory_config=MEM)
        embeddings = ttnn.add(embeddings, position_emb, memory_config=MEM)
        ttnn.deallocate(word_emb)
        ttnn.deallocate(token_type_emb)
        ttnn.deallocate(position_emb)

        embeddings = ttnn.layer_norm(
            embeddings,
            weight=self.parameters.embeddings.LayerNorm.weight,
            bias=self.parameters.embeddings.LayerNorm.bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_HIFI4,
            program_config=layernorm_program_config,
        )
        return embeddings

    def attention(self, hidden_states, attention_mask, layer_params, device):
        """Multi-head self-attention with fused QKV and sharded memory."""
        # Fused QKV projection
        qkv_output = ttnn.linear(
            hidden_states,
            layer_params.attention.self.query_key_value.weight,
            bias=layer_params.attention.self.query_key_value.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=qkv_program_config,
            dtype=ttnn.bfloat8_b,
        )

        # Split into Q, K, V and reshape to multi-head format
        (query, key, value) = ttnn.experimental.split_query_key_value_and_split_heads(
            qkv_output,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            num_heads=self.num_heads,
        )
        ttnn.deallocate(qkv_output)

        # Q * K^T
        attention_scores = ttnn.matmul(
            query,
            key,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=pre_softmax_program_config,
        )
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        # Fused mask + scale + softmax
        attention_probs = ttnn.transformer.attention_softmax_(
            attention_scores,
            attention_mask=attention_mask,
            head_size=self.head_dim,
            program_config=softmax_program_config,
        )

        # Attention * V
        context = ttnn.matmul(
            attention_probs,
            value,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(value)

        # Concat heads back
        context = ttnn.experimental.nlp_concat_heads(
            context,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        )

        # Output projection + residual + LayerNorm
        attn_output = ttnn.linear(
            context,
            layer_params.attention.output.dense.weight,
            bias=layer_params.attention.output.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=self_out_program_config,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(context)

        # Reshard residual to match output sharding, then fused residual + LN
        residual = ttnn.reshard(hidden_states, attn_output.memory_config())
        attn_output = ttnn.layer_norm(
            attn_output,
            residual_input_tensor=residual,
            weight=layer_params.attention.output.LayerNorm.weight,
            bias=layer_params.attention.output.LayerNorm.bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_HIFI4,
            program_config=layernorm_program_config,
        )
        return attn_output

    def ffn(self, hidden_states, layer_params):
        """FFN with sharded memory and fused GELU."""
        intermediate = ttnn.linear(
            hidden_states,
            layer_params.intermediate.dense.weight,
            bias=layer_params.intermediate.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=ff1_program_config,
            compute_kernel_config=COMPUTE_HIFI2,
            dtype=ttnn.bfloat8_b,
        )

        output = ttnn.linear(
            intermediate,
            layer_params.output.dense.weight,
            bias=layer_params.output.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=ff2_program_config,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(intermediate)

        # Reshard residual, then fused residual + LayerNorm
        residual = ttnn.reshard(hidden_states, output.memory_config())
        output = ttnn.layer_norm(
            output,
            residual_input_tensor=residual,
            weight=layer_params.output.LayerNorm.weight,
            bias=layer_params.output.LayerNorm.bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_HIFI4,
            program_config=layernorm_program_config,
        )
        return output

    def encoder_layer(self, hidden_states, attention_mask, layer_params, device):
        """Single transformer layer: attention + FFN."""
        attn_output = self.attention(hidden_states, attention_mask, layer_params, device)
        ttnn.deallocate(hidden_states)
        output = self.ffn(attn_output, layer_params)
        return output

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, device):
        """
        Forward pass with optimized sharded execution.

        Args:
            input_ids: [batch, seq] uint32
            token_type_ids: [batch, seq] uint32
            position_ids: [batch, seq] uint32
            attention_mask: [batch, 1, 1, seq] bfloat16 extended mask
            device: TT device

        Returns:
            last_hidden_state: [batch, 1, seq, 384] in L1 block-sharded memory
        """
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, device)

        for i in range(self.config.num_hidden_layers):
            hidden_states = self.encoder_layer(hidden_states, attention_mask, self.parameters.encoder.layer[i], device)

        return hidden_states
