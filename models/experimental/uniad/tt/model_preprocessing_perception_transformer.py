# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.uniad.reference.ffn import FFN
from models.experimental.uniad.reference.encoder import BEVFormerEncoder
from models.experimental.uniad.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.uniad.reference.spatial_cross_attention import SpatialCrossAttention

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, FFN):
        parameters = {
            "ffn": {},
        }
        parameters["ffn"][f"ffn0"] = {
            "linear1": {
                "weight": preprocess_linear_weight(model.layers[0][0].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.layers[0][0].bias, dtype=ttnn.bfloat16),
            },
            "linear2": {
                "weight": preprocess_linear_weight(model.layers[1].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.layers[1].bias, dtype=ttnn.bfloat16),
            },
        }

    if isinstance(model, BEVFormerEncoder):
        parameters = {"layers": {}}

        for i, layer in enumerate(model.layers):  # BaseTransformerLayer
            layer_dict = {
                "attentions": {},
                "ffn": {},
                "norms": {},
            }

            # Norms
            for n, norm in enumerate(layer.norms):
                if isinstance(norm, nn.LayerNorm):
                    layer_dict["norms"][f"norm{n}"] = {
                        "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                    }

            # FFNs
            for k, ffn in enumerate(layer.ffns):
                layer_dict["ffn"][f"ffn{k}"] = {
                    "linear1": {
                        "weight": preprocess_linear_weight(ffn.layers[0][0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[0][0].bias, dtype=ttnn.bfloat16),
                    },
                    "linear2": {
                        "weight": preprocess_linear_weight(ffn.layers[1].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[1].bias, dtype=ttnn.bfloat16),
                    },
                }

            # Attentions
            for j, attn in enumerate(layer.attentions):
                if isinstance(attn, TemporalSelfAttention):
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

                elif isinstance(attn, SpatialCrossAttention):
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(
                                attn.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(
                                attn.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16
                            ),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(
                                attn.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(
                                attn.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16
                            ),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(
                                attn.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(
                                attn.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16
                            ),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

            parameters["layers"][f"layer{i}"] = layer_dict

        return parameters

    return parameters


def create_uniad_model_parameters_perception_transformer(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


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
                layer_params[layer_index] = {
                    "weight": ttnn.to_device(preprocess_linear_weight(layer.weight, dtype=dtype), device=device),
                    "bias": ttnn.to_device(preprocess_linear_bias(layer.bias, dtype=dtype), device=device),
                }

            elif isinstance(layer, nn.LayerNorm):
                layer_params[layer_index] = {
                    "weight": preprocess_layernorm_parameter(layer.weight, dtype=dtype),
                    "bias": preprocess_layernorm_parameter(layer.bias, dtype=dtype),
                }
            layer_index += 1

        branch_params[i] = layer_params

    return branch_params
