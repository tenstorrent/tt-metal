# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.BEVFormerV2.reference.resnet import ResNet
from models.experimental.BEVFormerV2.reference.bevformer_v2 import BEVFormerV2
from models.experimental.BEVFormerV2.reference.fpn import FPN
from models.experimental.BEVFormerV2.reference.encoder import BEVFormerEncoder
from models.experimental.BEVFormerV2.reference.perception_transformer import PerceptionTransformerV2
from models.experimental.BEVFormerV2.reference.head import BEVFormerHead
from models.experimental.BEVFormerV2.reference.decoder import DetectionTransformerDecoder
from models.experimental.BEVFormerV2.reference.multihead_attention import (
    CustomMSDeformableAttention,
    MultiheadAttention,
)
from models.experimental.BEVFormerV2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.BEVFormerV2.reference.spatial_cross_attention import SpatialCrossAttention
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

    def extract_transformer_parameters(transformer_module):
        parameters = {"layers": {}}

        for i, layer in enumerate(transformer_module.layers):
            layer_dict = {
                "attentions": {},
                "ffn": {},
                "norms": {},
            }

            for n, norm in enumerate(getattr(layer, "norms", [])):
                if isinstance(norm, nn.LayerNorm):
                    layer_dict["norms"][f"norm{n}"] = {
                        "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                    }

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

    def extract_embeddings_to_ttnn(model, names, dtype):
        return {
            name: {"weight": ttnn.from_torch(getattr(model, name).weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)}
            for name in names
        }

    if isinstance(model, (BEVFormerV2, BEVFormerHead, DetectionTransformerDecoder, PerceptionTransformerV2)):
        if isinstance(model.pts_bbox_head, BEVFormerHead):
            head = model.pts_bbox_head
            parameters["head"] = {}

            parameters["head"]["positional_encoding"] = {}
            pos_encoding = head.positional_encoding
            parameters["head"]["positional_encoding"]["row_embed"] = {
                "weight": ttnn.from_torch(pos_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }
            parameters["head"]["positional_encoding"]["col_embed"] = {
                "weight": ttnn.from_torch(pos_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

            if isinstance(head.transformer, PerceptionTransformerV2):
                parameters["head"]["transformer"] = {}
                if isinstance(head.transformer.encoder, BEVFormerEncoder):
                    parameters["head"]["transformer"]["encoder"] = extract_transformer_parameters(
                        head.transformer.encoder
                    )

                if isinstance(head.transformer.decoder, DetectionTransformerDecoder):
                    parameters["head"]["transformer"]["decoder"] = extract_transformer_parameters(
                        head.transformer.decoder
                    )

                parameters["head"]["transformer"]["reference_points"] = {
                    "weight": preprocess_linear_weight(head.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(head.transformer.reference_points.bias, dtype=ttnn.bfloat16),
                }

                parameters["head"]["transformer"]["map_reference_points"] = {
                    "weight": preprocess_linear_weight(head.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(head.transformer.reference_points.bias, dtype=ttnn.bfloat16),
                }

                if hasattr(head.transformer, "level_embeds"):
                    parameters["head"]["transformer"]["level_embeds"] = ttnn.from_torch(
                        head.transformer.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                if hasattr(head.transformer, "cams_embeds"):
                    parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                        head.transformer.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )

            embedding_layers = ["bev_embedding", "query_embedding"]
            parameters["head"].update(extract_embeddings_to_ttnn(head, embedding_layers, dtype=ttnn.bfloat16))
            parameters["head"]["branches"] = {}

            parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
                head.cls_branches, dtype=ttnn.bfloat16
            )
            parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
                head.reg_branches, dtype=ttnn.bfloat16
            )

        if isinstance(model.img_neck, FPN):
            neck = model.img_neck
            parameters["img_neck"] = {}
            parameters["img_neck"]["lateral_convs"] = []
            for lateral_conv in neck.lateral_convs:
                conv = lateral_conv.conv
                lateral_params = {
                    "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                    if conv.bias is not None
                    else None,
                }
                parameters["img_neck"]["lateral_convs"].append(lateral_params)

            parameters["img_neck"]["fpn_convs"] = []
            for fpn_conv in neck.fpn_convs:
                conv = fpn_conv.conv
                fpn_params = {
                    "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                    if conv.bias is not None
                    else None,
                }
                parameters["img_neck"]["fpn_convs"].append(fpn_params)

        if isinstance(model.img_backbone, ResNet):
            backbone = model.img_backbone
            parameters["img_backbone"] = {}

            if isinstance(backbone.norm1, nn.Identity):
                parameters["img_backbone"]["conv1"] = {
                    "weight": ttnn.from_torch(backbone.conv1.weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(backbone.conv1.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                }
            else:
                weight, bias = fold_batch_norm2d_into_conv2d(backbone.conv1, backbone.norm1)
                parameters["img_backbone"]["conv1"] = {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                }

            for layer_idx in range(1, 5):
                layer = getattr(backbone, f"layer{layer_idx}")
                for block_idx, block in enumerate(layer):
                    prefix = f"layer{layer_idx}_{block_idx}"
                    parameters["img_backbone"][prefix] = {}

                    for conv_name in ["conv1", "conv2", "conv3"]:
                        conv = getattr(block, conv_name)
                        norm = getattr(block, f"norm{conv_name[-1]}")

                        if isinstance(norm, nn.Identity):
                            w = conv.weight
                            b = conv.bias
                        else:
                            w, b = fold_batch_norm2d_into_conv2d(conv, norm)

                        parameters["img_backbone"][prefix][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                    if hasattr(block, "downsample") and block.downsample is not None:
                        ds = block.downsample
                        if isinstance(ds, nn.Sequential):
                            conv = ds[0]
                            norm = ds[1]

                            if isinstance(norm, nn.Identity):
                                w = conv.weight
                                b = conv.bias
                            else:
                                w, b = fold_batch_norm2d_into_conv2d(conv, norm)

                            parameters["img_backbone"][prefix]["downsample"] = {
                                "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                                "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                            }

    return parameters


def create_bevformerv2_model_parameters(model: BEVFormerV2, input_tensor: input, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {"img_backbone": {}, "img_neck": {}}

    img = input_tensor[1]

    if isinstance(img, list):
        img = torch.tensor(img[0])
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)

    parameters.conv_args["img_backbone"] = infer_ttnn_module_args(
        model=model.img_backbone,
        run_model=lambda model: model(img),
        device=None,
    )

    img_feats = model.img_backbone(img)

    parameters.conv_args["img_neck"] = infer_ttnn_module_args(
        model=model.img_neck,
        run_model=lambda model: model(img_feats),
        device=None,
    )

    assert parameters is not None

    for key in parameters.conv_args.keys():
        if key == "img_backbone":
            for conv_key in parameters.conv_args[key].keys():
                if isinstance(conv_key, str) and hasattr(model.img_backbone, conv_key):
                    parameters.conv_args[key][conv_key].module = getattr(model.img_backbone, conv_key)
        elif key == "img_neck":
            for conv_key in parameters.conv_args[key].keys():
                if isinstance(conv_key, str) and hasattr(model.img_neck, conv_key):
                    parameters.conv_args[key][conv_key].module = getattr(model.img_neck, conv_key)

    if hasattr(parameters, "head"):
        parameters.head.cls_branches_torch = model.pts_bbox_head.cls_branches
        parameters.head.reg_branches_torch = model.pts_bbox_head.reg_branches

    return parameters
