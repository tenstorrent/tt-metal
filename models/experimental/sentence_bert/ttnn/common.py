# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

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
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device,
):
    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    attention_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return input_ids, token_type_ids, position_ids, attention_mask
