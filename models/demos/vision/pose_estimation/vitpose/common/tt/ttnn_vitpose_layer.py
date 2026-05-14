# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_attention import vitpose_attention
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_mlp import vitpose_mlp


def vitpose_layer(hidden_states, *, parameters, num_heads=12, compute_kernel_config=None):
    ln1 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_before.weight"],
        bias=parameters["layernorm_before.bias"],
        compute_kernel_config=compute_kernel_config,
    )

    attn_out = vitpose_attention(ln1, parameters=parameters, num_heads=num_heads, compute_kernel_config=compute_kernel_config)
    hidden_states = hidden_states + attn_out

    ln2 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_after.weight"],
        bias=parameters["layernorm_after.bias"],
        compute_kernel_config=compute_kernel_config,
    )

    mlp_out = vitpose_mlp(ln2, parameters=parameters, compute_kernel_config=compute_kernel_config)
    hidden_states = hidden_states + mlp_out

    return hidden_states


def preprocess_layer_parameters(state_dict, layer_idx, *, dtype=ttnn.bfloat16):
    """
    Preprocess all parameters for a single transformer layer.
    """
    from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_attention import (
        preprocess_attention_parameters,
    )
    from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_mlp import preprocess_mlp_parameters

    prefix = f"backbone.encoder.layer.{layer_idx}"
    params = {}

    for name in ["layernorm_before", "layernorm_after"]:
        w = state_dict[f"{prefix}.{name}.weight"].reshape(1, -1)
        b = state_dict[f"{prefix}.{name}.bias"].reshape(1, -1)
        params[f"{name}.weight"] = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        params[f"{name}.bias"] = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    attn_params = preprocess_attention_parameters(state_dict, layer_idx, dtype=dtype)
    params.update(attn_params)

    mlp_params = preprocess_mlp_parameters(state_dict, layer_idx, dtype=dtype)
    params.update(mlp_params)

    return params
