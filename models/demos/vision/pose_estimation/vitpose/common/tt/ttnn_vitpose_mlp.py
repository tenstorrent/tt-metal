# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def vitpose_mlp(hidden_states, *, parameters, compute_kernel_config=None):
    mm_kwargs = {}
    if compute_kernel_config is not None:
        mm_kwargs["compute_kernel_config"] = compute_kernel_config

    output = ttnn.matmul(hidden_states, parameters["fc1.weight"], **mm_kwargs)
    output = output + parameters["fc1.bias"]
    output = ttnn.gelu(output)
    output = ttnn.matmul(output, parameters["fc2.weight"], **mm_kwargs)
    output = output + parameters["fc2.bias"]
    return output


def preprocess_mlp_parameters(state_dict, layer_idx, *, dtype=ttnn.bfloat16):
    """
    Preprocess MLP parameters for a single layer from HuggingFace state dict.
    """
    prefix = f"backbone.encoder.layer.{layer_idx}.mlp"
    params = {}

    for name in ["fc1", "fc2"]:
        w = state_dict[f"{prefix}.{name}.weight"].T.contiguous()
        b = state_dict[f"{prefix}.{name}.bias"].reshape(1, -1)
        params[f"{name}.weight"] = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        params[f"{name}.bias"] = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    return params
