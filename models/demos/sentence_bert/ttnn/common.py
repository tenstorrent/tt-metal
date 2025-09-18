# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(6, 8),
    subblock_w=4,
    block_h=12,
    block_w=4,
    inplace=True,
)

ff1_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(6, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=8,
    per_core_M=12,
    per_core_N=16,
    transpose_mcast=False,
    fused_activation=(ttnn.UnaryOpType.GELU, True),
)

ff2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(6, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=False,
    fused_activation=None,
)

query_key_value_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(6, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=False,
    fused_activation=None,
)

self_out_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(6, 8),
    in0_block_w=4,
    out_subblock_h=2,
    out_subblock_w=4,
    per_core_M=12,
    per_core_N=4,
    transpose_mcast=False,
    fused_activation=None,
)
pre_softmax_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(6, 8),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=24,
    per_core_N=12,
)
softmax_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(6, 8),
    subblock_w=6,
    block_h=24,
    block_w=12,
)


def custom_preprocessor(torch_model, name):
    parameters = {}
    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        qw = torch_model.query.weight
        kw = torch_model.key.weight
        vw = torch_model.value.weight
        qb = torch_model.query.bias
        kb = torch_model.key.bias
        vb = torch_model.value.bias
        qw = torch.transpose(qw, -1, -2)
        kw = torch.transpose(kw, -1, -2)
        vw = torch.transpose(vw, -1, -2)
        const_w_dims = qw.shape[:-1]
        qw = qw.reshape([*const_w_dims, 6, -1])  # nums_attention_heads// 2
        kw = kw.reshape(qw.shape)
        vw = vw.reshape(qw.shape)
        qkv_weight_torch = torch.cat((qw, kw, vw), -1).reshape([*const_w_dims, -1])
        const_b_dims = qb.shape[:-1]
        qb = qb.reshape([*const_b_dims, 6, -1])  # nums_attention_heads// 2
        kb = kb.reshape(qb.shape)
        vb = vb.reshape(qb.shape)
        qkv_bias_torch = torch.cat((qb, kb, vb), -1).reshape([*const_b_dims, -1])

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = ttnn.from_torch(
            qkv_weight_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        parameters["query_key_value"]["bias"] = ttnn.from_torch(
            qkv_bias_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

    return parameters


def preprocess_inputs(
    input_ids=None,
    token_type_ids=None,
    position_ids=None,
    extended_attention_mask=None,
    attention_mask=None,
    device=None,
):
    if input_ids is not None:
        input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    if token_type_ids is not None:
        token_type_ids = ttnn.from_torch(
            token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    if position_ids is not None:
        position_ids = ttnn.from_torch(
            position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    if extended_attention_mask is not None:
        extended_attention_mask = ttnn.from_torch(
            extended_attention_mask,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    if attention_mask is not None:
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    return input_ids, token_type_ids, position_ids, extended_attention_mask, attention_mask


def ttnn_mean_pooling(ttnn_token_embeddings, ttnn_attention_mask, device=None):
    ttnn_token_embeddings = ttnn.sharded_to_interleaved(ttnn_token_embeddings, ttnn.L1_MEMORY_CONFIG)
    if ttnn_attention_mask.is_sharded():
        ttnn_attention_mask_interleaved = ttnn.sharded_to_interleaved(ttnn_attention_mask, ttnn.L1_MEMORY_CONFIG)
        ttnn_attention_mask_interleaved = ttnn.to_layout(ttnn_attention_mask_interleaved, ttnn.TILE_LAYOUT)
        ttnn.deallocate(ttnn_attention_mask)
    else:
        ttnn_attention_mask_interleaved = ttnn_attention_mask
    ttnn_token_embeddings = ttnn.squeeze(ttnn_token_embeddings, dim=1)
    tt_input_mask_expanded = ttnn.unsqueeze(ttnn_attention_mask_interleaved, dim=-1)
    tt_input_mask_expanded = ttnn.repeat(tt_input_mask_expanded, [1, 1, ttnn_token_embeddings.shape[-1]])
    sum1 = ttnn.multiply(ttnn_token_embeddings, tt_input_mask_expanded)
    sum1 = ttnn.sum(sum1, 1)
    sum2 = ttnn.sum(tt_input_mask_expanded, 1)
    sum2 = ttnn.clamp(sum2, min=1e-9)
    result = ttnn.div(sum1, sum2)
    return result
