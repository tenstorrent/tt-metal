# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn

from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


def extract_sequential_branch(module_list, dtype, device):
    branch_params = {}

    for i, mod in enumerate(module_list):
        layer_params = {}
        layer_index = 0

        if isinstance(mod, nn.Sequential):
            layers = mod
        elif hasattr(mod, "mlp") and isinstance(mod.mlp, nn.Sequential):
            layers = mod.mlp
        else:
            layers = [mod]

        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer_params[str(layer_index)] = {
                    "weight": ttnn.to_device(preprocess_linear_weight(layer.weight, dtype=dtype), device=device),
                    "bias": ttnn.to_device(preprocess_linear_bias(layer.bias, dtype=dtype), device=device),
                }
                layer_index += 1
            elif isinstance(layer, nn.LayerNorm):
                layer_params[f"{layer_index}_norm"] = {
                    "weight": preprocess_layernorm_parameter(layer.weight, dtype=dtype),
                    "bias": preprocess_layernorm_parameter(layer.bias, dtype=dtype),
                }
                layer_index += 1

        branch_params[str(i)] = layer_params

    return branch_params
