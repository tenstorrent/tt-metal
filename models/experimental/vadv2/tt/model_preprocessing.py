# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.vadv2.reference.resnet import ResNet
from models.experimental.vadv2.reference.fpn import FPN
from models.experimental.vadv2.reference.encoder import BEVFormerEncoder
from models.experimental.vadv2.reference.transformer import VADPerceptionTransformer
from models.experimental.vadv2.reference.head import VADHead

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
    fold_batch_norm2d_into_conv2d,
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
    if isinstance(model, ResNet):
        if isinstance(model, ResNet):
            parameters["res_model"] = {}

        # Initial conv + bn
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}_{block_idx}"
                parameters["res_model"][prefix] = {}

                # conv1, conv2, conv3
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    bn = getattr(block, f"bn{conv_name[-1]}")
                    w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                    parameters["res_model"][prefix][conv_name] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

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

    def extract_single_linears(model, layer_names, dtype):
        params = {}
        for name in layer_names:
            layer = getattr(model, name)
            params[name] = {
                "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
            }
        return params

    def extract_positional_encoding(model, dtype):
        pos_encoding = model.positional_encoding

        return {
            "row_embed": {
                "weight": ttnn.from_torch(pos_encoding.row_embed.weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            },
            "col_embed": {
                "weight": ttnn.from_torch(pos_encoding.col_embed.weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            },
        }

    def extract_embeddings_to_ttnn(model, names, dtype):
        return {
            name: {"weight": ttnn.from_torch(getattr(model, name).weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)}
            for name in names
        }

    if isinstance(model, (VADHead, CustomTransformerDecoder, VADPerceptionTransformer)):
        parameters = {}
        parameters["head"] = {}

        parameters["head"]["positional_encoding"] = {}
        pos_encoding = model.positional_encoding
        parameters["head"]["positional_encoding"]["row_embed"] = {
            "weight": ttnn.from_torch(pos_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        }
        parameters["head"]["positional_encoding"]["col_embed"] = {
            "weight": ttnn.from_torch(pos_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        }

        if isinstance(model.motion_decoder, CustomTransformerDecoder):
            parameters["head"]["motion_decoder"] = extract_transformer_parameters(model.motion_decoder)
        if isinstance(model.motion_map_decoder, CustomTransformerDecoder):
            parameters["head"]["motion_map_decoder"] = extract_transformer_parameters(model.motion_map_decoder)
        if isinstance(model.ego_map_decoder, CustomTransformerDecoder):
            parameters["head"]["ego_map_decoder"] = extract_transformer_parameters(model.ego_map_decoder)
        if isinstance(model.ego_agent_decoder, CustomTransformerDecoder):
            parameters["head"]["ego_agent_decoder"] = extract_transformer_parameters(model.ego_agent_decoder)

        if isinstance(model.transformer, VADPerceptionTransformer):
            parameters["head"]["transformer"] = {}
            if isinstance(model.transformer.encoder, BEVFormerEncoder):
                parameters["head"]["transformer"]["encoder"] = extract_transformer_parameters(model.transformer.encoder)

            if isinstance(model.transformer.decoder, DetectionTransformerDecoder):
                print("Executedddddddddddd")
                parameters["head"]["transformer"]["decoder"] = extract_transformer_parameters(model.transformer.decoder)

            # Handle map_decoder if present
            if isinstance(model.transformer.map_decoder, MapDetectionTransformerDecoder):
                print("yes here")
                parameters["head"]["transformer"]["map_decoder"] = extract_transformer_parameters(
                    model.transformer.map_decoder
                )
            single_linear_layers = ["pos_mlp_sa", "pos_mlp", "ego_agent_pos_mlp", "ego_map_pos_mlp"]
            parameters["head"].update(extract_single_linears(model, single_linear_layers, ttnn.bfloat16))

            parameters["head"]["transformer"]["reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.reference_points.bias, dtype=ttnn.bfloat16),
            }

            parameters["head"]["transformer"]["map_reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.map_reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.map_reference_points.bias, dtype=ttnn.bfloat16),
            }

            # CAN Bus MLP
            parameters["head"]["transformer"]["can_bus_mlp"] = {
                "0": {
                    "weight": preprocess_linear_weight(model.transformer.can_bus_mlp[0].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.transformer.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
                },
                "1": {
                    "weight": preprocess_linear_weight(model.transformer.can_bus_mlp[2].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.transformer.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
                },
                "norm": {
                    "weight": preprocess_layernorm_parameter(
                        model.transformer.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16
                    ),
                    "bias": preprocess_layernorm_parameter(
                        model.transformer.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16
                    ),
                },
            }

            parameters["head"]["transformer"]["level_embeds"] = ttnn.from_torch(
                model.transformer.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                model.transformer.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        embedding_layers = [
            "bev_embedding",
            "query_embedding",
            "map_instance_embedding",
            "map_pts_embedding",
            "motion_mode_query",
            "ego_query",
        ]
        parameters["head"].update(extract_embeddings_to_ttnn(model, embedding_layers, dtype=ttnn.bfloat16))
        parameters["head"]["branches"] = {}

        parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
            model.cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
            model.reg_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["traj_branches"] = extract_sequential_branch(
            model.traj_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["traj_cls_branches"] = extract_sequential_branch(
            model.traj_cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["map_cls_branches"] = extract_sequential_branch(
            model.map_cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["map_reg_branches"] = extract_sequential_branch(
            model.map_reg_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["ego_fut_decoder"] = extract_sequential_branch(
            model.ego_fut_decoder, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["agent_fus_mlp"] = extract_sequential_branch(
            model.agent_fus_mlp, dtype=ttnn.bfloat16
        )

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


def create_vadv2_model_parameters_head(model: ResNet, device=None):
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


def create_vadv2_model_parameters(model: ResNet, input_tensor, device=None):
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
