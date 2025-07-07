# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.vadv2.reference.resnet import ResNet
from models.experimental.vadv2.reference.fpn import FPN
from models.experimental.vadv2.reference.encoder import BEVFormerEncoder
from models.experimental.vadv2.reference.transformer import VADPerceptionTransformer

from models.experimental.vadv2.reference.decoder import (
    CustomTransformerDecoder,
    DetectionTransformerDecoder,
    MapDetectionTransformerDecoder,
    CustomMSDeformableAttention,
    MultiheadAttention,
)
from models.experimental.vadv2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.vadv2.reference.spatial_cross_attention import SpatialCrossAttention


from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, FPN):
        parameters["fpn"] = {}
        parameters["fpn"]["lateral_convs"] = {}
        parameters["fpn"]["lateral_convs"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs.conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs.conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        parameters["fpn"]["fpn_convs"] = {}
        parameters["fpn"]["fpn_convs"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs.conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs.conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    def extract_transformer_parameters(transformer_module):
        parameters = {"layers": {}}

        for i, layer in enumerate(transformer_module.layers):  # BaseTransformerLayer
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
                            "weight": preprocess_linear_weight(
                                deform_attn.sampling_offsets.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(deform_attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(
                                deform_attn.attention_weights.weight, dtype=ttnn.bfloat16
                            ),
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

    if isinstance(model, (CustomTransformerDecoder, VADPerceptionTransformer)):
        parameters = {}
        # parameters = parameters[""]
        # print(model.encoder)
        # ss

        print("helloo")
        if isinstance(model, CustomTransformerDecoder):
            parameters["motion_decoder"] = extract_transformer_parameters(model)
        else:
            if isinstance(model.encoder, BEVFormerEncoder):
                parameters["encoder"] = extract_transformer_parameters(model.encoder)

            if isinstance(model.decoder, DetectionTransformerDecoder):
                print("Executedddddddddddd")
                parameters["decoder"] = extract_transformer_parameters(model.decoder)

            # Handle map_decoder if present
            if isinstance(model.map_decoder, MapDetectionTransformerDecoder):
                print("yes here")
                parameters["map_decoder"] = extract_transformer_parameters(model.map_decoder)

            parameters["reference_points"] = {
                "weight": preprocess_linear_weight(model.reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.reference_points.bias, dtype=ttnn.bfloat16),
            }

            parameters["map_reference_points"] = {
                "weight": preprocess_linear_weight(model.map_reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.map_reference_points.bias, dtype=ttnn.bfloat16),
            }

            # CAN Bus MLP
            parameters["can_bus_mlp"] = {
                "0": {
                    "weight": preprocess_linear_weight(model.can_bus_mlp[0].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
                },
                "1": {
                    "weight": preprocess_linear_weight(model.can_bus_mlp[2].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
                },
                "norm": {
                    "weight": preprocess_layernorm_parameter(model.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_layernorm_parameter(model.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16),
                },
            }

            parameters["level_embeds"] = ttnn.from_torch(
                model.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            parameters["cams_embeds"] = ttnn.from_torch(model.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    return parameters


def create_vadv2_model_parameters(model: ResNet, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


def create_vadv2_model_parameters_sca(model: ResNet, input_tensor, device=None):
    print("Pre process modle parameters")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(
            input_tensor[0],
            key=input_tensor[1],
            value=input_tensor[2],
            reference_points=input_tensor[3],
            spatial_shapes=input_tensor[4],
            reference_points_cam=input_tensor[5],
            bev_mask=input_tensor[6],
            level_start_index=input_tensor[7],
        ),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


def create_vadv2_model_parameters_tsa(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(
            input_tensor[0],
            query_pos=input_tensor[1],
            reference_points=input_tensor[2],
            spatial_shapes=input_tensor[3],
            level_start_index=input_tensor[4],
        ),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


def create_vadv2_model_parameters_decoder(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


def create_vadv2_model_parameters_encoder(model: ResNet, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters
