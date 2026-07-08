# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.uniad.reference.motion_head import MotionHead
from models.experimental.uniad.reference.fpn import FPN
from models.experimental.uniad.reference.ffn import FFN
from models.experimental.uniad.reference.resnet import ResNet, ModulatedDeformConv2dPack
from models.experimental.uniad.reference.encoder import BEVFormerEncoder
from models.experimental.uniad.reference.decoder import (
    MultiheadAttention,
    CustomMSDeformableAttention,
)
from models.experimental.uniad.reference.motion_transformer_decoder import (
    MotionDeformableAttention,
    MotionTransformerAttentionLayer,
)

from models.experimental.uniad.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.uniad.reference.spatial_cross_attention import SpatialCrossAttention
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    infer_ttnn_module_args,
    preprocess_layernorm_parameter,
    fold_batch_norm2d_into_conv2d,
)
from models.experimental.uniad.reference.motion_transformer_decoder import (
    MapInteraction,
    TrackAgentInteraction,
    IntentionInteraction,
)


def extract_sequential_branch(module_list, dtype):
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
                    "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                    "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
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


def custom_preprocessor_bev(model, name):
    parameters = {}

    if isinstance(model, MotionDeformableAttention):
        parameters["sampling_offsets"] = {}
        parameters["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.sampling_offsets.bias, dtype=ttnn.bfloat16
        )

        parameters["attention_weights"] = {}
        parameters["attention_weights"]["weight"] = preprocess_linear_weight(
            model.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["attention_weights"]["bias"] = preprocess_linear_bias(
            model.attention_weights.bias, dtype=ttnn.bfloat16
        )

        parameters["value_proj"] = {}
        parameters["value_proj"]["weight"] = preprocess_linear_weight(model.value_proj.weight, dtype=ttnn.bfloat16)
        parameters["value_proj"]["bias"] = preprocess_linear_bias(model.value_proj.bias, dtype=ttnn.bfloat16)

        parameters["output_proj"] = {}
        parameters["output_proj"][0] = {}
        parameters["output_proj"][0]["weight"] = preprocess_linear_weight(
            model.output_proj[0].weight, dtype=ttnn.bfloat16
        )
        parameters["output_proj"][0]["bias"] = preprocess_linear_bias(model.output_proj[0].bias, dtype=ttnn.bfloat16)

        parameters["output_proj"][1] = {}
        parameters["output_proj"][1]["weight"] = preprocess_layernorm_parameter(
            model.output_proj[1].weight, dtype=ttnn.bfloat16
        )
        parameters["output_proj"][1]["bias"] = preprocess_layernorm_parameter(
            model.output_proj[1].bias, dtype=ttnn.bfloat16
        )

    if isinstance(model, MotionTransformerAttentionLayer):
        parameters["attentions"] = {}
        for i, child in enumerate(model.attentions):
            parameters_deformable_attention = {}
            if isinstance(child, MotionDeformableAttention):
                parameters_deformable_attention["sampling_offsets"] = {}
                parameters_deformable_attention["sampling_offsets"]["weight"] = preprocess_linear_weight(
                    child.sampling_offsets.weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["sampling_offsets"]["bias"] = preprocess_linear_bias(
                    child.sampling_offsets.bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["attention_weights"] = {}
                parameters_deformable_attention["attention_weights"]["weight"] = preprocess_linear_weight(
                    child.attention_weights.weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["attention_weights"]["bias"] = preprocess_linear_bias(
                    child.attention_weights.bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["value_proj"] = {}
                parameters_deformable_attention["value_proj"]["weight"] = preprocess_linear_weight(
                    child.value_proj.weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["value_proj"]["bias"] = preprocess_linear_bias(
                    child.value_proj.bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["output_proj"] = {}
                parameters_deformable_attention["output_proj"][0] = {}
                parameters_deformable_attention["output_proj"][0]["weight"] = preprocess_linear_weight(
                    child.output_proj[0].weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["output_proj"][0]["bias"] = preprocess_linear_bias(
                    child.output_proj[0].bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["output_proj"][1] = {}
                parameters_deformable_attention["output_proj"][1]["weight"] = preprocess_layernorm_parameter(
                    child.output_proj[1].weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["output_proj"][1]["bias"] = preprocess_layernorm_parameter(
                    child.output_proj[1].bias, dtype=ttnn.bfloat16
                )
            parameters["attentions"][i] = parameters_deformable_attention

        parameters["ffns"] = {}

        for i, ffn in enumerate(model.ffns):
            if isinstance(ffn, FFN):
                parameters["ffns"][f"{i}"] = {
                    "linear1": {
                        "weight": preprocess_linear_weight(ffn.layers[0][0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[0][0].bias, dtype=ttnn.bfloat16),
                    },
                    "linear2": {
                        "weight": preprocess_linear_weight(ffn.layers[1].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[1].bias, dtype=ttnn.bfloat16),
                    },
                }

        parameters["norms"] = {}

        parameters["norms"][0] = {}
        parameters["norms"][0]["weight"] = preprocess_layernorm_parameter(model.norms[0].weight, dtype=ttnn.bfloat16)

        parameters["norms"][0]["bias"] = preprocess_layernorm_parameter(model.norms[1].bias, dtype=ttnn.bfloat16)

        parameters["norms"][1] = {}
        parameters["norms"][1]["weight"] = preprocess_layernorm_parameter(model.norms[1].weight, dtype=ttnn.bfloat16)

        parameters["norms"][1]["bias"] = preprocess_layernorm_parameter(model.norms[1].bias, dtype=ttnn.bfloat16)

    return parameters


def extract_transformer_parameters(transformer_module):
    parameters = {"layers": {}}

    for i, layer in enumerate(transformer_module.layers):
        layer_dict = {
            "attentions": {},
            "ffn": {},
            "norms": {},
        }

        # ---- Norms ----
        for n, norm in enumerate(getattr(layer, "norms", [])):
            if isinstance(norm, nn.LayerNorm):
                layer_dict["norms"][f"norm{n}"] = {
                    "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                }

        # ---- FFNs ----
        for k, ffn in enumerate(getattr(layer, "ffns", [])):
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

        # ---- Attentions ----
        for j, attn in enumerate(getattr(layer, "attentions", [])):
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
                deform_attn = attn.deformable_attention
                layer_dict["attentions"][f"attn{j}"] = {
                    "sampling_offsets": {
                        "weight": preprocess_linear_weight(deform_attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(deform_attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                    },
                    "attention_weights": {
                        "weight": preprocess_linear_weight(deform_attn.attention_weights.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(deform_attn.attention_weights.bias, dtype=ttnn.bfloat16),
                    },
                    "value_proj": {
                        "weight": preprocess_linear_weight(deform_attn.value_proj.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(deform_attn.value_proj.bias, dtype=ttnn.bfloat16),
                    },
                    "output_proj": {
                        "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                    },
                }

            elif isinstance(attn, MultiheadAttention):
                layer_dict["attentions"][f"attn{j}"] = {
                    "in_proj": {
                        "weight": preprocess_linear_weight(attn.attn.in_proj_weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(attn.attn.in_proj_bias, dtype=ttnn.bfloat16),
                    },
                    "out_proj": {
                        "weight": preprocess_linear_weight(attn.attn.out_proj.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(attn.attn.out_proj.bias, dtype=ttnn.bfloat16),
                    },
                }

            elif isinstance(attn, CustomMSDeformableAttention):
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

        parameters["layers"][f"layer{i}"] = layer_dict
    return parameters


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

    if isinstance(model, ResNet):
        backbone = model

        # Initial conv + bn
        weight, bias = fold_batch_norm2d_into_conv2d(backbone.conv1, backbone.bn1)
        parameters["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(backbone, f"layer{layer_idx}")
            prefix = f"layer{layer_idx}"
            parameters[prefix] = {}
            for block_idx, block in enumerate(layer):
                parameters[prefix][block_idx] = {}

                # conv1, conv2, conv3
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    if isinstance(conv, ModulatedDeformConv2dPack):
                        parameters[prefix][block_idx][conv_name] = {}
                        parameters[prefix][block_idx][conv_name]["weight"] = conv.weight
                        parameters[prefix][block_idx][conv_name]["bias"] = conv.bias
                        parameters[prefix][block_idx][conv_name]["conv_offset"] = {
                            "weight": ttnn.from_torch(conv.conv_offset.weight, dtype=ttnn.bfloat16),
                            "bias": ttnn.from_torch(conv.conv_offset.bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
                        }

                        bn = getattr(block, f"bn{conv_name[-1]}")
                        channel_size = bn.num_features

                        # Extract PyTorch tensors
                        weight_torch = bn.weight if bn.affine else None
                        bias_torch = bn.bias if bn.affine else None
                        batch_mean_torch = bn.running_mean
                        batch_var_torch = bn.running_var

                        # Reshape for broadcast compatibility (1, C, 1, 1)
                        batch_mean_torch = batch_mean_torch.view(1, channel_size, 1, 1)
                        batch_var_torch = batch_var_torch.view(1, channel_size, 1, 1)
                        weight_torch = weight_torch.view(1, channel_size, 1, 1) if weight_torch is not None else None
                        bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias_torch is not None else None

                        parameters[prefix][block_idx]["bn2"] = {}
                        weight = (
                            ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if weight_torch is not None
                            else None
                        )
                        parameters[prefix][block_idx]["bn2"]["weight"] = weight  # ttnn.to_device(weight, device)

                        bias = (
                            ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if bias_torch is not None
                            else None
                        )
                        parameters[prefix][block_idx]["bn2"]["bias"] = bias  # ttnn.to_device(bias, device)

                        running_mean = ttnn.from_torch(batch_mean_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters[prefix][block_idx]["bn2"]["running_mean"] = running_mean

                        running_var = ttnn.from_torch(batch_var_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters[prefix][block_idx]["bn2"]["running_var"] = running_var

                        parameters[prefix][block_idx]["bn2"]["eps"] = bn.eps  # scalar, used directly in ops

                    else:
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters[prefix][block_idx][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.bfloat16),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
                        }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters[prefix][block_idx]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.bfloat16),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
                        }

    if isinstance(model, FPN):
        neck = model

        # Lateral Convs
        parameters["fpn"] = {}

        # Lateral Convs
        parameters["fpn"]["lateral_convs"] = {}

        parameters["fpn"]["lateral_convs"]["0"] = {}
        parameters["fpn"]["lateral_convs"]["0"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[0].conv.weight, dtype=ttnn.bfloat16
        )
        bias = model.lateral_convs[0].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["height"] = 80
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["width"] = 45
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["batch"] = 6

        parameters["fpn"]["lateral_convs"]["1"] = {}
        parameters["fpn"]["lateral_convs"]["1"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[1].conv.weight, dtype=ttnn.bfloat16
        )
        bias = model.lateral_convs[1].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["height"] = 40
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["width"] = 23
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["batch"] = 6

        parameters["fpn"]["lateral_convs"]["2"] = {}
        parameters["fpn"]["lateral_convs"]["2"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[2].conv.weight, dtype=ttnn.bfloat16
        )
        bias = model.lateral_convs[2].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["height"] = 20
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["width"] = 12
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["batch"] = 6
        # FPN Convs
        parameters["fpn"]["fpn_convs"] = {}

        parameters["fpn"]["fpn_convs"]["0"] = {}
        parameters["fpn"]["fpn_convs"]["0"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[0].conv.weight, dtype=ttnn.bfloat16
        )
        bias = model.fpn_convs[0].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["height"] = 80
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["width"] = 45
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["1"] = {}
        parameters["fpn"]["fpn_convs"]["1"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[1].conv.weight, dtype=ttnn.bfloat16
        )
        bias = model.fpn_convs[1].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["height"] = 40
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["width"] = 23
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["2"] = {}
        parameters["fpn"]["fpn_convs"]["2"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[2].conv.weight, dtype=ttnn.bfloat16
        )
        bias = model.fpn_convs[2].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["height"] = 20
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["width"] = 12
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["3"] = {}
        parameters["fpn"]["fpn_convs"]["3"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[3].conv.weight, dtype=ttnn.bfloat16
        )
        bias = model.fpn_convs[3].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["height"] = 20
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["width"] = 12
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["batch"] = 6

        parameters["model_args"] = neck  # For conv configs

    if isinstance(model, MotionHead):
        motion_head = model
        parameters["learnable_motion_query_embedding"] = {}
        parameters["learnable_motion_query_embedding"]["weight"] = ttnn.from_torch(
            motion_head.learnable_motion_query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        parameters["motionformer"] = custom_preprocessor_motion_decoder(motion_head.motionformer, None)

        parameters["layer_track_query_fuser"] = {}
        parameters["layer_track_query_fuser"][0] = {}
        parameters["layer_track_query_fuser"][0]["weight"] = preprocess_linear_weight(
            motion_head.layer_track_query_fuser[0].weight, dtype=ttnn.bfloat16
        )
        parameters["layer_track_query_fuser"][0]["bias"] = preprocess_linear_bias(
            motion_head.layer_track_query_fuser[0].bias, dtype=ttnn.bfloat16
        )

        parameters["layer_track_query_fuser"][1] = {}
        parameters["layer_track_query_fuser"][1]["weight"] = preprocess_layernorm_parameter(
            motion_head.layer_track_query_fuser[1].weight, dtype=ttnn.bfloat16
        )
        parameters["layer_track_query_fuser"][1]["bias"] = preprocess_layernorm_parameter(
            motion_head.layer_track_query_fuser[1].bias, dtype=ttnn.bfloat16
        )

        parameters["layer_track_query_fuser"][2] = {}

        parameters["agent_level_embedding_layer"] = custom_preprocessor_layer(
            motion_head.agent_level_embedding_layer, None
        )
        parameters["scene_level_ego_embedding_layer"] = custom_preprocessor_layer(
            motion_head.scene_level_ego_embedding_layer, None
        )
        parameters["scene_level_offset_embedding_layer"] = custom_preprocessor_layer(
            motion_head.scene_level_offset_embedding_layer, None
        )
        parameters["boxes_query_embedding_layer"] = custom_preprocessor_layer(
            motion_head.boxes_query_embedding_layer, None
        )

        parameters["traj_cls_branches"] = {}
        for index, child in enumerate(motion_head.traj_cls_branches):
            parameters_tmp = {}

            parameters_tmp[0] = {}
            parameters_tmp[0]["weight"] = preprocess_linear_weight(child[0].weight, dtype=ttnn.bfloat16)
            parameters_tmp[0]["bias"] = preprocess_linear_bias(child[0].bias, dtype=ttnn.bfloat16)

            parameters_tmp[1] = {}
            parameters_tmp[1]["weight"] = preprocess_layernorm_parameter(child[1].weight, dtype=ttnn.bfloat16)
            parameters_tmp[1]["bias"] = preprocess_layernorm_parameter(child[1].bias, dtype=ttnn.bfloat16)
            parameters_tmp[2] = {}

            parameters_tmp[3] = {}
            parameters_tmp[3]["weight"] = preprocess_linear_weight(child[3].weight, dtype=ttnn.bfloat16)
            parameters_tmp[3]["bias"] = preprocess_linear_bias(child[3].bias, dtype=ttnn.bfloat16)

            parameters_tmp[4] = {}
            parameters_tmp[4]["weight"] = preprocess_layernorm_parameter(child[4].weight, dtype=ttnn.bfloat16)
            parameters_tmp[4]["bias"] = preprocess_layernorm_parameter(child[4].bias, dtype=ttnn.bfloat16)
            parameters_tmp[5] = {}

            parameters_tmp[6] = {}
            parameters_tmp[6]["weight"] = preprocess_linear_weight(child[6].weight, dtype=ttnn.bfloat16)
            parameters_tmp[6]["bias"] = preprocess_linear_bias(child[6].bias, dtype=ttnn.bfloat16)

            parameters["traj_cls_branches"][index] = parameters_tmp

        parameters["traj_reg_branches"] = {}
        for index, child in enumerate(motion_head.traj_reg_branches):
            parameters_temp = {}

            parameters_temp[0] = {}
            parameters_temp[0]["weight"] = preprocess_linear_weight(child[0].weight, dtype=ttnn.bfloat16)
            parameters_temp[0]["bias"] = preprocess_linear_bias(child[0].bias, dtype=ttnn.bfloat16)

            parameters_temp[1] = {}

            parameters_temp[2] = {}
            parameters_temp[2]["weight"] = preprocess_linear_weight(child[2].weight, dtype=ttnn.bfloat16)
            parameters_temp[2]["bias"] = preprocess_linear_bias(child[2].bias, dtype=ttnn.bfloat16)

            parameters_temp[3] = {}

            parameters_temp[4] = {}
            parameters_temp[4]["weight"] = preprocess_linear_weight(child[4].weight, dtype=ttnn.bfloat16)
            parameters_temp[4]["bias"] = preprocess_linear_bias(child[4].bias, dtype=ttnn.bfloat16)

            parameters["traj_reg_branches"][index] = parameters_temp

    if isinstance(model, nn.Conv2d):
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

    return parameters


def custom_preprocessor_layer(child, name):
    parameters_tmp = {}
    parameters_tmp[0] = {}
    parameters_tmp[0]["weight"] = preprocess_linear_weight(child[0].weight, dtype=ttnn.bfloat16)
    parameters_tmp[0]["bias"] = preprocess_linear_bias(child[0].bias, dtype=ttnn.bfloat16)

    parameters_tmp[1] = {}

    parameters_tmp[2] = {}
    parameters_tmp[2]["weight"] = preprocess_linear_weight(child[2].weight, dtype=ttnn.bfloat16)
    parameters_tmp[2]["bias"] = preprocess_linear_bias(child[2].bias, dtype=ttnn.bfloat16)

    return parameters_tmp


def custom_preprocessor_interaction(model, name):
    parameters = {}
    if isinstance(model, MapInteraction) or isinstance(model, TrackAgentInteraction):
        parameters["interaction_transformer"] = {}
        child = model.interaction_transformer
        if isinstance(child, nn.TransformerDecoderLayer):
            parameters_tmp = {}

            parameters_tmp["self_attn"] = {}
            parameters_tmp["self_attn"]["in_proj_weight"] = ttnn.from_torch(
                child.self_attn.in_proj_weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["in_proj_bias"] = ttnn.from_torch(
                child.self_attn.in_proj_bias, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"] = {}
            parameters_tmp["self_attn"]["out_proj"]["weight"] = preprocess_linear_weight(
                child.self_attn.out_proj.weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"]["bias"] = preprocess_linear_bias(
                child.self_attn.out_proj.bias, dtype=ttnn.bfloat16
            )

            parameters_tmp["multihead_attn"] = {}
            parameters_tmp["multihead_attn"]["in_proj_weight"] = ttnn.from_torch(
                child.multihead_attn.in_proj_weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["multihead_attn"]["in_proj_bias"] = ttnn.from_torch(
                child.multihead_attn.in_proj_bias, dtype=ttnn.bfloat16
            )
            parameters_tmp["multihead_attn"]["out_proj"] = {}
            parameters_tmp["multihead_attn"]["out_proj"]["weight"] = preprocess_linear_weight(
                child.multihead_attn.out_proj.weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["multihead_attn"]["out_proj"]["bias"] = preprocess_linear_bias(
                child.multihead_attn.out_proj.bias, dtype=ttnn.bfloat16
            )

            parameters_tmp["linear1"] = {}
            parameters_tmp["linear1"]["weight"] = preprocess_linear_weight(child.linear1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear1"]["bias"] = preprocess_linear_bias(child.linear1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["linear2"] = {}
            parameters_tmp["linear2"]["weight"] = preprocess_linear_weight(child.linear2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear2"]["bias"] = preprocess_linear_bias(child.linear2.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm1"] = {}
            parameters_tmp["norm1"]["weight"] = preprocess_layernorm_parameter(child.norm1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm1"]["bias"] = preprocess_layernorm_parameter(child.norm1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm2"] = {}
            parameters_tmp["norm2"]["weight"] = preprocess_layernorm_parameter(child.norm2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm2"]["bias"] = preprocess_layernorm_parameter(child.norm2.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm3"] = {}
            parameters_tmp["norm3"]["weight"] = preprocess_layernorm_parameter(child.norm3.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm3"]["bias"] = preprocess_layernorm_parameter(child.norm3.bias, dtype=ttnn.bfloat16)

        parameters["interaction_transformer"] = parameters_tmp

    if isinstance(model, IntentionInteraction):
        parameters["interaction_transformer"] = {}
        child = model.interaction_transformer
        if isinstance(child, nn.TransformerEncoderLayer):
            parameters_tmp = {}
            parameters_tmp["self_attn"] = {}
            parameters_tmp["self_attn"]["in_proj_weight"] = ttnn.from_torch(
                child.self_attn.in_proj_weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["in_proj_bias"] = ttnn.from_torch(
                child.self_attn.in_proj_bias, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"] = {}
            parameters_tmp["self_attn"]["out_proj"]["weight"] = preprocess_linear_weight(
                child.self_attn.out_proj.weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"]["bias"] = preprocess_linear_bias(
                child.self_attn.out_proj.bias, dtype=ttnn.bfloat16
            )

            parameters_tmp["linear1"] = {}
            parameters_tmp["linear1"]["weight"] = preprocess_linear_weight(child.linear1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear1"]["bias"] = preprocess_linear_bias(child.linear1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["linear2"] = {}
            parameters_tmp["linear2"]["weight"] = preprocess_linear_weight(child.linear2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear2"]["bias"] = preprocess_linear_bias(child.linear2.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm1"] = {}
            parameters_tmp["norm1"]["weight"] = preprocess_layernorm_parameter(child.norm1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm1"]["bias"] = preprocess_layernorm_parameter(child.norm1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm2"] = {}
            parameters_tmp["norm2"]["weight"] = preprocess_layernorm_parameter(child.norm2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm2"]["bias"] = preprocess_layernorm_parameter(child.norm2.bias, dtype=ttnn.bfloat16)

        parameters["interaction_transformer"] = parameters_tmp

    return parameters


def custom_preprocessor_motion_decoder(model, name):
    parameters = {}
    parameters["intention_interaction_layers"] = custom_preprocessor_interaction(
        model.intention_interaction_layers, None
    )

    parameters["track_agent_interaction_layers"] = {}
    for index, child in enumerate(model.track_agent_interaction_layers):
        parameters["track_agent_interaction_layers"][index] = custom_preprocessor_interaction(child, None)

    parameters["map_interaction_layers"] = {}
    for index, child in enumerate(model.map_interaction_layers):
        parameters["map_interaction_layers"][index] = custom_preprocessor_interaction(child, None)

    parameters["bev_interaction_layers"] = {}
    for index, child in enumerate(model.bev_interaction_layers):
        parameters["bev_interaction_layers"][index] = custom_preprocessor_bev(child, None)

    parameters["static_dynamic_fuser"] = {}
    parameters["static_dynamic_fuser"][0] = {}

    parameters["static_dynamic_fuser"][0]["weight"] = preprocess_linear_weight(
        model.static_dynamic_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["static_dynamic_fuser"][0]["bias"] = preprocess_linear_bias(
        model.static_dynamic_fuser[0].bias, dtype=ttnn.bfloat16
    )

    parameters["static_dynamic_fuser"][2] = {}
    parameters["static_dynamic_fuser"][2]["weight"] = preprocess_linear_weight(
        model.static_dynamic_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["static_dynamic_fuser"][2]["bias"] = preprocess_linear_bias(
        model.static_dynamic_fuser[2].bias, dtype=ttnn.bfloat16
    )

    parameters["dynamic_embed_fuser"] = {}
    parameters["dynamic_embed_fuser"][0] = {}
    parameters["dynamic_embed_fuser"][0]["weight"] = preprocess_linear_weight(
        model.dynamic_embed_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["dynamic_embed_fuser"][0]["bias"] = preprocess_linear_bias(
        model.dynamic_embed_fuser[0].bias, dtype=ttnn.bfloat16
    )

    parameters["dynamic_embed_fuser"][2] = {}
    parameters["dynamic_embed_fuser"][2]["weight"] = preprocess_linear_weight(
        model.dynamic_embed_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["dynamic_embed_fuser"][2]["bias"] = preprocess_linear_bias(
        model.dynamic_embed_fuser[2].bias, dtype=ttnn.bfloat16
    )

    parameters["in_query_fuser"] = {}
    parameters["in_query_fuser"][0] = {}
    parameters["in_query_fuser"][0]["weight"] = preprocess_linear_weight(
        model.in_query_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["in_query_fuser"][0]["bias"] = preprocess_linear_bias(model.in_query_fuser[0].bias, dtype=ttnn.bfloat16)

    parameters["in_query_fuser"][2] = {}
    parameters["in_query_fuser"][2]["weight"] = preprocess_linear_weight(
        model.in_query_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["in_query_fuser"][2]["bias"] = preprocess_linear_bias(model.in_query_fuser[2].bias, dtype=ttnn.bfloat16)

    parameters["out_query_fuser"] = {}
    parameters["out_query_fuser"][0] = {}
    parameters["out_query_fuser"][0]["weight"] = preprocess_linear_weight(
        model.out_query_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["out_query_fuser"][0]["bias"] = preprocess_linear_bias(
        model.out_query_fuser[0].bias, dtype=ttnn.bfloat16
    )

    parameters["out_query_fuser"][2] = {}
    parameters["out_query_fuser"][2]["weight"] = preprocess_linear_weight(
        model.out_query_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["out_query_fuser"][2]["bias"] = preprocess_linear_bias(
        model.out_query_fuser[2].bias, dtype=ttnn.bfloat16
    )

    return parameters


def _prepare_one_conv_in_place(
    param_node,
    conv_args_node,
    conv_module,
    device,
    torch_weight,
    torch_bias,
    weights_dtype=ttnn.bfloat8_b,
    shard_layout=None,
    activation=None,
):
    """Replace the float32 raw weight/bias in `param_node` with TTNN-prepared
    versions, eliminating conv2d's first-call 'pull back to host' warning.

    Takes torch_weight/torch_bias directly (not pulled back from device) to
    avoid an extra D2H+H2D round-trip during prep. `weights_dtype` and
    `shard_layout` must match the Conv2dConfig the consuming
    TtConv2d/TtnnConv2D wrapper uses at runtime — mismatch leads to wrong
    sharding spec and a giant allocator request at runtime.
    """
    has_bias = torch_bias is not None

    conv_config_kwargs = {"weights_dtype": weights_dtype}
    if shard_layout is not None:
        conv_config_kwargs["shard_layout"] = shard_layout
    if activation is not None:
        conv_config_kwargs["activation"] = activation

    common_kwargs = dict(
        # L1 (not DRAM): the runtime TtnnConv2D wrapper uses Conv2dConfig(shard_layout=HEIGHT_SHARDED)
        # and lets the op manage its own activations. Passing DRAM here drives setup_conv_prep_config
        # down the is_dram_conv branch, which constructs a dummy weight TensorSpec with
        # BFLOAT8_B + Layout::ROW_MAJOR — a combo the new (2026-05-22) TensorSpec validator rejects.
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        in_channels=conv_module.in_channels,
        out_channels=conv_module.out_channels,
        batch_size=conv_args_node.batch_size,
        input_height=conv_args_node.input_height,
        input_width=conv_args_node.input_width,
        kernel_size=tuple(conv_module.kernel_size),
        stride=tuple(conv_module.stride),
        padding=tuple(conv_module.padding),
        dilation=tuple(conv_module.dilation),
        groups=conv_module.groups,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=ttnn.Conv2dConfig(**conv_config_kwargs),
    )
    # NOTE: feed prepare_conv_weights the raw torch weight as float32 so its
    # internal layout/dtype conversion sees the full precision input (it will
    # produce a properly-tiled bf8_b weight per conv_config). Feeding lossy
    # input dtypes (bf16/bf8_b) here was measured to noticeably degrade the
    # first ttnn forward.
    param_node.weight = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_weight, dtype=ttnn.float32),
        weights_format="OIHW",
        has_bias=has_bias,
        **common_kwargs,
    )
    if has_bias:
        bias_for_prep = torch_bias.reshape((1, 1, 1, -1)) if torch_bias.ndim == 1 else torch_bias
        param_node.bias = ttnn.prepare_conv_bias(
            bias_tensor=ttnn.from_torch(bias_for_prep, dtype=ttnn.float32),
            **common_kwargs,
        )


def _prepare_resnet_backbone_convs(parameters, model, device):
    """Pre-prepare every nn.Conv2d weight in ResNet-101 (skipping DCN which is
    a host fallback). Eliminates ~200 of the ~234 conv2d 'not properly
    prepared' warnings produced on the first e2e inference."""
    backbone = model.img_backbone
    bp = parameters.img_backbone
    ca = bp.conv_args

    # Stem: conv1 is folded with bn1.
    w, b = fold_batch_norm2d_into_conv2d(backbone.conv1, backbone.bn1)
    _prepare_one_conv_in_place(bp.conv1, ca.conv1, backbone.conv1, device, w, b)

    # layer1..layer4 (Bottleneck blocks)
    for li in range(1, 5):
        layer_key = f"layer{li}"
        ca_layer = ca[layer_key]
        bp_layer = bp[layer_key]
        torch_layer = getattr(backbone, layer_key)
        for block_idx, torch_block in enumerate(torch_layer):
            bp_block = bp_layer[block_idx]
            ca_block = ca_layer[block_idx]
            for conv_name in ("conv1", "conv2", "conv3"):
                torch_conv = getattr(torch_block, conv_name)
                if isinstance(torch_conv, ModulatedDeformConv2dPack):
                    # DCN itself is a host fallback; pre-prep the conv_offset
                    # sub-conv (which DOES use ttnn.conv2d). conv_offset is
                    # unfolded — pass its raw weight/bias.
                    if conv_name in ca_block and "conv_offset" in ca_block[conv_name]:
                        _prepare_one_conv_in_place(
                            bp_block[conv_name].conv_offset,
                            ca_block[conv_name].conv_offset,
                            torch_conv.conv_offset,
                            device,
                            torch_conv.conv_offset.weight,
                            torch_conv.conv_offset.bias,
                        )
                    continue
                if conv_name not in ca_block:
                    continue
                # Bottleneck convN is folded with bnN.
                bn = getattr(torch_block, f"bn{conv_name[-1]}")
                w, b = fold_batch_norm2d_into_conv2d(torch_conv, bn)
                _prepare_one_conv_in_place(bp_block[conv_name], ca_block[conv_name], torch_conv, device, w, b)
            # downsample (if present). conv_args.downsample is a Sequential,
            # so the Conv2d entry is keyed by its Sequential index (0).
            if (
                hasattr(torch_block, "downsample")
                and torch_block.downsample is not None
                and "downsample" in bp_block
                and "downsample" in ca_block
            ):
                torch_ds_conv = torch_block.downsample[0]
                torch_ds_bn = torch_block.downsample[1]
                w, b = fold_batch_norm2d_into_conv2d(torch_ds_conv, torch_ds_bn)
                ds_ca = ca_block.downsample
                ds_conv_args = ds_ca[0] if 0 in ds_ca else ds_ca["0"]
                if "batch_size" in ds_conv_args:
                    _prepare_one_conv_in_place(bp_block.downsample, ds_conv_args, torch_ds_conv, device, w, b)


def _conv_args_dict(
    *, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, batch_size, input_height, input_width
):
    """Build a simple namespace-like dict that matches the attribute access in
    `_prepare_one_conv_in_place` for cases where we don't have a real
    conv_args entry from infer_ttnn_module_args (e.g., FPN, bev_adapter)."""

    class _Ns:
        pass

    ns = _Ns()
    ns.in_channels = in_channels
    ns.out_channels = out_channels
    ns.kernel_size = kernel_size
    ns.stride = stride
    ns.padding = padding
    ns.dilation = dilation
    ns.groups = groups
    ns.batch_size = batch_size
    ns.input_height = input_height
    ns.input_width = input_width
    return ns


def _prepare_fpn_convs(parameters, model, device):
    """Pre-prepare FPN lateral and FPN-output Conv2d weights/bias. The input
    spatial dimensions are already encoded in the parameters tree as
    height/width/batch."""
    fpn = model.img_neck
    bp_fpn = parameters.img_neck.fpn

    for group_name, group_module in (("lateral_convs", fpn.lateral_convs), ("fpn_convs", fpn.fpn_convs)):
        bp_group = bp_fpn[group_name]
        for idx, conv_block in enumerate(group_module):
            torch_conv = conv_block.conv  # ConvModule wraps a nn.Conv2d at .conv
            bp_node = bp_group[str(idx)].conv
            ca = _conv_args_dict(
                in_channels=torch_conv.in_channels,
                out_channels=torch_conv.out_channels,
                kernel_size=tuple(torch_conv.kernel_size),
                stride=tuple(torch_conv.stride),
                padding=tuple(torch_conv.padding),
                dilation=tuple(torch_conv.dilation),
                groups=torch_conv.groups,
                batch_size=bp_node["batch"],
                input_height=bp_node["height"],
                input_width=bp_node["width"],
            )
            _prepare_one_conv_in_place(bp_node, ca, torch_conv, device, torch_conv.weight, torch_conv.bias)


def _prepare_bev_adapter_convs(parameters, model, device):
    """Pre-prepare PlanningHead bev_adapter's 6 Conv2d.

    bev_adapter = nn.Sequential of 3 blocks, each block = nn.Sequential(
        Conv2d(256->128, 3x3, pad=1), ReLU, Conv2d(128->256, 1x1)
    )
    Input to bev_adapter is the BEV grid (50x50, batch=1) and the spatial size
    is preserved through every conv (3x3 with padding=1; 1x1).

    Different from ResNet/FPN: bev_adapter uses TtConv2d (in ttnn_planning_head)
    which defaults to weights_dtype=bfloat16. The prep dtype must match.
    """
    bev_adapter = model.planning_head.bev_adapter
    bp_bev = parameters.planning_head.bev_adapter
    BEV_H, BEV_W, BATCH = 50, 50, 1
    relu_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    for block_idx in range(len(bev_adapter)):
        block = bev_adapter[block_idx]
        bp_block = bp_bev[block_idx]
        for conv_local_idx in (0, 2):  # 0 = 3x3 (+RELU), 2 = 1x1 (no activation)
            torch_conv = block[conv_local_idx]
            bp_node = bp_block[conv_local_idx].conv
            ca = _conv_args_dict(
                in_channels=torch_conv.in_channels,
                out_channels=torch_conv.out_channels,
                kernel_size=tuple(torch_conv.kernel_size),
                stride=tuple(torch_conv.stride),
                padding=tuple(torch_conv.padding),
                dilation=tuple(torch_conv.dilation),
                groups=torch_conv.groups,
                batch_size=BATCH,
                input_height=BEV_H,
                input_width=BEV_W,
            )
            # TtConv2d (planning_head) defaults: shard_layout=HEIGHT_SHARDED,
            # activation=RELU on the 3x3 conv (idx 0), activation=None on
            # the 1x1 conv (idx 2). Mirror these in prep to keep weight
            # spec aligned with the runtime path.
            _prepare_one_conv_in_place(
                bp_node,
                ca,
                torch_conv,
                device,
                torch_conv.weight,
                torch_conv.bias,
                weights_dtype=ttnn.bfloat16,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                activation=relu_activation if conv_local_idx == 0 else None,
            )


def create_uniad_model_parameters_uniad(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["planning_head"]["model_args"] = model.planning_head
    parameters["img_backbone"].conv_args = {}
    parameters.img_backbone.conv_args = infer_ttnn_module_args(
        model=model.img_backbone, run_model=lambda model: model(torch.randn(6, 3, 640, 360)), device=None
    )
    assert parameters.img_backbone is not None
    for key in parameters.img_backbone.conv_args.keys():
        parameters.img_backbone.conv_args[key].module = getattr(model.img_backbone, key)

    # Pre-prepare ResNet + FPN Conv2d weights so the first-call "pulling back
    # to host" reprep is avoided. The bev_adapter prep helper is kept in this
    # file but not called — it currently triggers a 5.5 TB DRAM allocation at
    # planning_head.forward_test (matching the runtime Conv2dConfig
    # shard_layout / activation flags still produced wrong weight specs).
    # Tracking as a follow-up.
    _prepare_resnet_backbone_convs(parameters, model, device)
    _prepare_fpn_convs(parameters, model, device)
    return parameters
