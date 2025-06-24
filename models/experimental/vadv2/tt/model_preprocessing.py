# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.vadv2.reference.resnet import ResNet
from models.experimental.vadv2.reference.fpn import FPN

from models.experimental.vadv2.reference.decoder import (
    DetectionTransformerDecoder,
    MapDetectionTransformerDecoder,
    CustomMSDeformableAttention,
    MultiheadAttention,
)
from models.experimental.vadv2.reference.temporal_self_attention import TemporalSelfAttention


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

    if isinstance(model, TemporalSelfAttention):
        parameters["temporal_self_attention"] = {}
        parameters["temporal_self_attention"]["sampling_offsets"] = {}
        parameters["temporal_self_attention"]["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.sampling_offsets.bias, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["attention_weights"] = {}
        parameters["temporal_self_attention"]["attention_weights"]["weight"] = preprocess_linear_weight(
            model.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["attention_weights"]["bias"] = preprocess_linear_bias(
            model.attention_weights.bias, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["value_proj"] = {}
        parameters["temporal_self_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
            model.value_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
            model.value_proj.bias, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["output_proj"] = {}
        parameters["temporal_self_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
            model.output_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
            model.output_proj.bias, dtype=ttnn.bfloat16
        )
    # if isinstance(model, MultiheadAttention):
    #     print("dtghbnnjnm", model.attn.in_proj_weight.shape)
    #     parameters = {
    #         "attentions": {},
    #     }
    #     parameters["attentions"][f"attn{0}"] = {
    #         "in_proj": {
    #             "weight": preprocess_linear_weight(model.attn.in_proj_weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.attn.in_proj_bias, dtype=ttnn.bfloat16),
    #         },
    #         "out_proj": {
    #             "weight": preprocess_linear_weight(model.attn.out_proj.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.attn.out_proj.bias, dtype=ttnn.bfloat16),
    #         },
    #     }
    # if isinstance(model, FFN):
    #     parameters = {
    #         "ffn": {},
    #     }
    #     parameters["ffn"][f"ffn0"] = {
    #             "linear1": {
    #                 "weight": preprocess_linear_weight(model.layers[0][0].weight, dtype=ttnn.bfloat16),
    #                 "bias": preprocess_linear_bias(model.layers[0][0].bias, dtype=ttnn.bfloat16),
    #             },
    #             "linear2": {
    #                 "weight": preprocess_linear_weight(model.layers[1].weight, dtype=ttnn.bfloat16),
    #                 "bias": preprocess_linear_bias(model.layers[1].bias, dtype=ttnn.bfloat16),
    #             },
    #         }

    # if isinstance(model, CustomMSDeformableAttention):
    #     parameters = {
    #         "attentions": {},
    #     }
    #     parameters["attentions"][f"attn1"] = {
    #         "sampling_offsets": {
    #             "weight": preprocess_linear_weight(model.sampling_offsets.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.sampling_offsets.bias, dtype=ttnn.bfloat16),
    #         },
    #         "attention_weights": {
    #             "weight": preprocess_linear_weight(model.attention_weights.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.attention_weights.bias, dtype=ttnn.bfloat16),
    #         },
    #         "value_proj": {
    #             "weight": preprocess_linear_weight(model.value_proj.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.value_proj.bias, dtype=ttnn.bfloat16),
    #         },
    #         "output_proj": {
    #             "weight": preprocess_linear_weight(model.output_proj.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.output_proj.bias, dtype=ttnn.bfloat16),
    #         },
    #     }

    if isinstance(model, MapDetectionTransformerDecoder or DetectionTransformerDecoder):
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
                if isinstance(attn, MultiheadAttention):
                    print(attn.attn.in_proj_weight.shape)
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

    # if isinstance(model, SpatialCrossAttention):
    #     parameters["spatial_cross_attention"] = {}
    #     parameters["spatial_cross_attention"]["deformable_attention"] = {}
    #     parameters["spatial_cross_attention"]["deformable_attention"]["sampling_offsets"] = {}
    #     parameters["spatial_cross_attention"]["deformable_attention"]["sampling_offsets"][
    #         "weight"
    #     ] = preprocess_linear_weight(model.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16)
    #     parameters["spatial_cross_attention"]["deformable_attention"]["sampling_offsets"][
    #         "bias"
    #     ] = preprocess_linear_bias(model.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16)
    #     parameters["spatial_cross_attention"]["deformable_attention"]["attention_weights"] = {}
    #     parameters["spatial_cross_attention"]["deformable_attention"]["attention_weights"][
    #         "weight"
    #     ] = preprocess_linear_weight(model.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16)
    #     parameters["spatial_cross_attention"]["deformable_attention"]["attention_weights"][
    #         "bias"
    #     ] = preprocess_linear_bias(model.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16)
    #     parameters["spatial_cross_attention"]["deformable_attention"]["value_proj"] = {}
    #     parameters["spatial_cross_attention"]["deformable_attention"]["value_proj"][
    #         "weight"
    #     ] = preprocess_linear_weight(model.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16)
    #     parameters["spatial_cross_attention"]["deformable_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
    #         model.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16
    #     )
    #     parameters["spatial_cross_attention"]["output_proj"] = {}
    #     parameters["spatial_cross_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
    #         model.output_proj.weight, dtype=ttnn.bfloat16
    #     )
    #     parameters["spatial_cross_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
    #         model.output_proj.bias, dtype=ttnn.bfloat16
    #     )

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
