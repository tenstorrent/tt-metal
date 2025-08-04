# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.uniad.reference.motion_head import MotionHead
from models.experimental.uniad.reference.fpn import FPN
from models.experimental.uniad.reference.resnet import ResNet, ModulatedDeformConv2dPack
from models.experimental.uniad.reference.encoder import BEVFormerEncoder
from models.experimental.uniad.reference.decoder import (
    DetectionTransformerDecoder,
    MultiheadAttention,
    CustomMSDeformableAttention,
)
from models.experimental.uniad.reference.head import BEVFormerTrackHead
from models.experimental.uniad.reference.perception_transformer import PerceptionTransformer
from models.experimental.uniad.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.uniad.reference.spatial_cross_attention import SpatialCrossAttention
from models.experimental.uniad.reference.uniad import UniAD
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
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

    if isinstance(model, (UniAD)):
        if isinstance(model.pts_bbox_head, BEVFormerTrackHead):
            bev_head = model.pts_bbox_head
            parameters["bev_head"] = {}
            parameters["bev_head"]["positional_encoding"] = {}
            pos_encoding = bev_head.positional_encoding
            parameters["bev_head"]["positional_encoding"]["row_embed"] = {
                "weight": ttnn.from_torch(pos_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }
            parameters["bev_head"]["positional_encoding"]["col_embed"] = {
                "weight": ttnn.from_torch(pos_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }
            if isinstance(bev_head.transformer, PerceptionTransformer):
                parameters["bev_head"]["transformer"] = {}
                if isinstance(bev_head.transformer.encoder, BEVFormerEncoder):
                    parameters["bev_head"]["transformer"]["encoder"] = extract_transformer_parameters(
                        bev_head.transformer.encoder
                    )

                if isinstance(bev_head.transformer.decoder, DetectionTransformerDecoder):
                    parameters["bev_head"]["transformer"]["decoder"] = extract_transformer_parameters(
                        bev_head.transformer.decoder
                    )
                parameters["bev_head"]["transformer"]["can_bus_mlp"] = {
                    "0": {
                        "weight": preprocess_linear_weight(
                            bev_head.transformer.can_bus_mlp[0].weight, dtype=ttnn.bfloat16
                        ),
                        "bias": preprocess_linear_bias(bev_head.transformer.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
                    },
                    "1": {
                        "weight": preprocess_linear_weight(
                            bev_head.transformer.can_bus_mlp[2].weight, dtype=ttnn.bfloat16
                        ),
                        "bias": preprocess_linear_bias(bev_head.transformer.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
                    },
                    "norm": {
                        "weight": preprocess_layernorm_parameter(
                            bev_head.transformer.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16
                        ),
                        "bias": preprocess_layernorm_parameter(
                            bev_head.transformer.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16
                        ),
                    },
                }
            parameters["bev_head"]["branches"] = {}
            parameters["bev_head"]["branches"]["cls_branches"] = extract_sequential_branch(
                bev_head.cls_branches, dtype=ttnn.bfloat16
            )
            parameters["bev_head"]["branches"]["reg_branches"] = extract_sequential_branch(
                bev_head.reg_branches, dtype=ttnn.bfloat16
            )
            parameters["bev_head"]["branches"]["past_traj_reg_branches"] = extract_sequential_branch(
                bev_head.reg_branches, dtype=ttnn.bfloat16
            )
        if isinstance(model.img_backbone, ResNet):
            backbone = model.img_backbone
            parameters["img_backbone"] = {}

            # Initial conv + bn
            weight, bias = fold_batch_norm2d_into_conv2d(backbone.conv1, backbone.bn1)
            parameters["img_backbone"]["conv1"] = {
                "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
                "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
            }

            # Loop over all layers (layer1 to layer4)
            for layer_idx in range(1, 5):
                layer = getattr(backbone, f"layer{layer_idx}")
                prefix = f"layer{layer_idx}"  # _{block_idx}"
                parameters["img_backbone"][prefix] = {}
                for block_idx, block in enumerate(layer):
                    parameters["img_backbone"][prefix][block_idx] = {}

                    # conv1, conv2, conv3
                    for conv_name in ["conv1", "conv2", "conv3"]:
                        conv = getattr(block, conv_name)
                        if isinstance(conv, ModulatedDeformConv2dPack):
                            parameters["img_backbone"][prefix][block_idx][conv_name] = {}
                            parameters["img_backbone"][prefix][block_idx][conv_name]["weight"] = conv.weight
                            parameters["img_backbone"][prefix][block_idx][conv_name]["bias"] = conv.bias
                            parameters["img_backbone"][prefix][block_idx][conv_name]["conv_offset"] = {
                                "weight": ttnn.from_torch(conv.conv_offset.weight, dtype=ttnn.float32),
                                "bias": ttnn.from_torch(
                                    conv.conv_offset.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32
                                ),
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
                            weight_torch = (
                                weight_torch.view(1, channel_size, 1, 1) if weight_torch is not None else None
                            )
                            bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias_torch is not None else None

                            parameters["img_backbone"][prefix][block_idx]["bn2"] = {}
                            weight = (
                                ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                                if weight_torch is not None
                                else None
                            )
                            parameters["img_backbone"][prefix][block_idx]["bn2"][
                                "weight"
                            ] = weight  # ttnn.to_device(weight, device)

                            bias = (
                                ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                                if bias_torch is not None
                                else None
                            )
                            parameters["img_backbone"][prefix][block_idx]["bn2"][
                                "bias"
                            ] = bias  # ttnn.to_device(bias, device)

                            running_mean = ttnn.from_torch(
                                batch_mean_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                            )
                            parameters["img_backbone"][prefix][block_idx]["bn2"][
                                "running_mean"
                            ] = running_mean  # ttnn.to_device(running_mean, device)

                            running_var = ttnn.from_torch(batch_var_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            parameters["img_backbone"][prefix][block_idx]["bn2"]["running_var"] = running_var

                            parameters["img_backbone"][prefix][block_idx]["bn2"][
                                "eps"
                            ] = bn.eps  # scalar, used directly in ops

                        else:
                            bn = getattr(block, f"bn{conv_name[-1]}")
                            w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                            parameters["img_backbone"][prefix][block_idx][conv_name] = {
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
                            parameters["img_backbone"][prefix][block_idx]["downsample"] = {
                                "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                                "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                            }

        if isinstance(model.img_neck, FPN):
            neck = model.img_neck
            parameters["img_neck"] = {}

            # Lateral Convs
            parameters["img_neck"]["lateral_convs"] = {}

            parameters["img_neck"]["lateral_convs"]["0"] = {}
            parameters["img_neck"]["lateral_convs"]["0"]["conv"] = {}
            parameters["img_neck"]["lateral_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
                neck.lateral_convs[0].conv.weight, dtype=ttnn.float32
            )
            bias = neck.lateral_convs[0].conv.bias.reshape((1, 1, 1, -1))
            parameters["img_neck"]["lateral_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
            parameters["img_neck"]["lateral_convs"]["0"]["conv"]["height"] = 80
            parameters["img_neck"]["lateral_convs"]["0"]["conv"]["width"] = 45
            parameters["img_neck"]["lateral_convs"]["0"]["conv"]["batch"] = 6

            parameters["img_neck"]["lateral_convs"]["1"] = {}
            parameters["img_neck"]["lateral_convs"]["1"]["conv"] = {}
            parameters["img_neck"]["lateral_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
                neck.lateral_convs[1].conv.weight, dtype=ttnn.float32
            )
            bias = neck.lateral_convs[1].conv.bias.reshape((1, 1, 1, -1))
            parameters["img_neck"]["lateral_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
            parameters["img_neck"]["lateral_convs"]["1"]["conv"]["height"] = 40
            parameters["img_neck"]["lateral_convs"]["1"]["conv"]["width"] = 23
            parameters["img_neck"]["lateral_convs"]["1"]["conv"]["batch"] = 6

            parameters["img_neck"]["lateral_convs"]["2"] = {}
            parameters["img_neck"]["lateral_convs"]["2"]["conv"] = {}
            parameters["img_neck"]["lateral_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
                neck.lateral_convs[2].conv.weight, dtype=ttnn.float32
            )
            bias = neck.lateral_convs[2].conv.bias.reshape((1, 1, 1, -1))
            parameters["img_neck"]["lateral_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
            parameters["img_neck"]["lateral_convs"]["2"]["conv"]["height"] = 20
            parameters["img_neck"]["lateral_convs"]["2"]["conv"]["width"] = 12
            parameters["img_neck"]["lateral_convs"]["2"]["conv"]["batch"] = 6
            # FPN Convs
            parameters["img_neck"]["fpn_convs"] = {}

            parameters["img_neck"]["fpn_convs"]["0"] = {}
            parameters["img_neck"]["fpn_convs"]["0"]["conv"] = {}
            parameters["img_neck"]["fpn_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
                neck.fpn_convs[0].conv.weight, dtype=ttnn.float32
            )
            bias = neck.fpn_convs[0].conv.bias.reshape((1, 1, 1, -1))
            parameters["img_neck"]["fpn_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
            parameters["img_neck"]["fpn_convs"]["0"]["conv"]["height"] = 80
            parameters["img_neck"]["fpn_convs"]["0"]["conv"]["width"] = 45
            parameters["img_neck"]["fpn_convs"]["0"]["conv"]["batch"] = 6

            parameters["img_neck"]["fpn_convs"]["1"] = {}
            parameters["img_neck"]["fpn_convs"]["1"]["conv"] = {}
            parameters["img_neck"]["fpn_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
                neck.fpn_convs[1].conv.weight, dtype=ttnn.float32
            )
            bias = neck.fpn_convs[1].conv.bias.reshape((1, 1, 1, -1))
            parameters["img_neck"]["fpn_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
            parameters["img_neck"]["fpn_convs"]["1"]["conv"]["height"] = 80
            parameters["img_neck"]["fpn_convs"]["1"]["conv"]["width"] = 45
            parameters["img_neck"]["fpn_convs"]["1"]["conv"]["batch"] = 6

            parameters["img_neck"]["fpn_convs"]["2"] = {}
            parameters["img_neck"]["fpn_convs"]["2"]["conv"] = {}
            parameters["img_neck"]["fpn_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
                neck.fpn_convs[2].conv.weight, dtype=ttnn.float32
            )
            bias = neck.fpn_convs[2].conv.bias.reshape((1, 1, 1, -1))
            parameters["img_neck"]["fpn_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
            parameters["img_neck"]["fpn_convs"]["2"]["conv"]["height"] = 40
            parameters["img_neck"]["fpn_convs"]["2"]["conv"]["width"] = 23
            parameters["img_neck"]["fpn_convs"]["2"]["conv"]["batch"] = 6

        if isinstance(model.motion_head, MotionHead):
            motion_head = model.motion_head
            parameters["motion_head"] = {}
            parameters["motion_head"]["learnable_motion_query_embedding"] = {}
            parameters["motion_head"]["learnable_motion_query_embedding"]["weight"] = ttnn.from_torch(
                motion_head.learnable_motion_query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

            parameters["motion_head"]["motionformer"] = custom_preprocessor_motion_decoder(
                motion_head.motionformer, None
            )

            parameters["motion_head"]["layer_track_query_fuser"] = {}
            parameters["motion_head"]["layer_track_query_fuser"][0] = {}
            parameters["motion_head"]["layer_track_query_fuser"][0]["weight"] = preprocess_linear_weight(
                motion_head.layer_track_query_fuser[0].weight, dtype=ttnn.bfloat16
            )
            parameters["motion_head"]["layer_track_query_fuser"][0]["bias"] = preprocess_linear_bias(
                motion_head.layer_track_query_fuser[0].bias, dtype=ttnn.bfloat16
            )

            parameters["motion_head"]["layer_track_query_fuser"][1] = {}
            parameters["motion_head"]["layer_track_query_fuser"][1]["weight"] = preprocess_layernorm_parameter(
                motion_head.layer_track_query_fuser[1].weight, dtype=ttnn.bfloat16
            )
            parameters["motion_head"]["layer_track_query_fuser"][1]["bias"] = preprocess_layernorm_parameter(
                motion_head.layer_track_query_fuser[1].bias, dtype=ttnn.bfloat16
            )

            parameters["motion_head"]["layer_track_query_fuser"][2] = {}

            # for layer in ["agent_level_embedding_layer","scene_level_ego_embedding_layer","scene_level_offset_embedding_layer","boxes_query_embedding_layer"]:
            parameters["motion_head"]["agent_level_embedding_layer"] = custom_preprocessor_layer(
                motion_head.agent_level_embedding_layer, None
            )
            parameters["motion_head"]["scene_level_ego_embedding_layer"] = custom_preprocessor_layer(
                motion_head.scene_level_ego_embedding_layer, None
            )
            parameters["motion_head"]["scene_level_offset_embedding_layer"] = custom_preprocessor_layer(
                motion_head.scene_level_offset_embedding_layer, None
            )
            parameters["motion_head"]["boxes_query_embedding_layer"] = custom_preprocessor_layer(
                motion_head.boxes_query_embedding_layer, None
            )

            parameters["motion_head"]["traj_cls_branches"] = {}
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

                parameters["motion_head"]["traj_cls_branches"][index] = parameters_tmp

            parameters["motion_head"]["traj_reg_branches"] = {}
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

                parameters["motion_head"]["traj_reg_branches"][index] = parameters_temp

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
            # parameters_tmp["self_attn"] = child.self_attn
            # parameters_tmp["multihead_attn"] = child.multihead_attn

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
            parameters_tmp["self_attn"] = child.self_attn

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
        parameters["bev_interaction_layers"][index] = custom_preprocessor(child, None)

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


def create_uniad_model_parameters_perception_transformer(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters
