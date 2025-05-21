# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from ttnn.model_preprocessing import (
    preprocess_linear_bias,
    preprocess_linear_weight,
)

layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    subblock_w=3,
    block_h=12,
    block_w=3,
    inplace=True,
)

ff1_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=3,
    out_subblock_h=1,
    out_subblock_w=8,
    per_core_M=12,
    per_core_N=16,
    transpose_mcast=True,
    fused_activation=(ttnn.UnaryOpType.GELU, True),
)

ff2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=True,
    fused_activation=None,
)

query_key_value_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=3,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=True,
    fused_activation=None,
)


def custom_preprocessor(torch_model, name):
    parameters = {}
    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        qkv_weight = torch.cat(
            [
                torch_model.query.weight,
                torch_model.key.weight,
                torch_model.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.query.bias, torch_model.key.bias, torch_model.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)

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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return input_ids, token_type_ids, position_ids, attention_mask
