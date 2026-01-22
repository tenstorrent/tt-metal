# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.MapTR.reference.maptr import (
    MapTR,
    MapTRHead,
    MapTRPerceptionTransformer,
    MapTRDecoder,
)
from models.experimental.MapTR.reference.bevformer import (
    BEVFormerEncoder,
    DetectionTransformerDecoder,
    CustomMSDeformableAttention,
    TemporalSelfAttention,
    SpatialCrossAttention,
)
from models.experimental.MapTR.reference.dependency import ResNet, FPN


from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
    fold_batch_norm2d_into_conv2d,
)


def custom_preprocessor(model, name):
    """Custom preprocessor for MapTR model parameters."""
    parameters = {}

    def extract_transformer_parameters(transformer_module):
        """Extract parameters from transformer encoder/decoder layers."""
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
                # FFN layers structure: [Linear, Activation, Dropout, Linear, Dropout]
                # First Linear is at index 0, second Linear is at index 3
                layer_dict["ffn"][f"ffn{k}"] = {
                    "linear1": {
                        "weight": preprocess_linear_weight(ffn.layers[0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[0].bias, dtype=ttnn.bfloat16),
                    },
                    "linear2": {
                        "weight": preprocess_linear_weight(ffn.layers[3].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[3].bias, dtype=ttnn.bfloat16),
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

                elif hasattr(attn, "attn"):
                    # MultiheadAttention
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

            parameters["layers"][f"layer{i}"] = layer_dict
        return parameters

    def extract_sequential_branch(module_list, dtype):
        """Extract parameters from sequential branches (cls_branches, reg_branches)."""
        branch_params = {}

        for i, mod in enumerate(module_list):
            layer_params = {}
            layer_index = 0

            if isinstance(mod, nn.Sequential):
                layers = mod
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

    def extract_positional_encoding(pos_encoding, dtype):
        """Extract positional encoding parameters."""
        return {
            "row_embed": {
                "weight": ttnn.from_torch(pos_encoding.row_embed.weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            },
            "col_embed": {
                "weight": ttnn.from_torch(pos_encoding.col_embed.weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            },
        }

    def extract_embeddings_to_ttnn(model, names, dtype):
        """Extract embedding weights to TTNN tensors."""
        embeddings = {}
        for name in names:
            if hasattr(model, name):
                embedding = getattr(model, name)
                if embedding is not None:
                    embeddings[name] = {
                        "weight": ttnn.from_torch(embedding.weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)
                    }
        return embeddings

    # Process MapTR model
    if isinstance(model, MapTR):
        # Process head
        if hasattr(model, "pts_bbox_head") and isinstance(model.pts_bbox_head, MapTRHead):
            head = model.pts_bbox_head
            parameters["head"] = {}

            # Positional encoding
            if hasattr(head, "positional_encoding") and head.positional_encoding is not None:
                parameters["head"]["positional_encoding"] = extract_positional_encoding(
                    head.positional_encoding, ttnn.bfloat16
                )

            # Query embeddings
            embedding_layers = ["bev_embedding", "instance_embedding", "pts_embedding", "query_embedding"]
            parameters["head"].update(extract_embeddings_to_ttnn(head, embedding_layers, dtype=ttnn.bfloat16))

            # Transformer
            if hasattr(head, "transformer") and isinstance(head.transformer, MapTRPerceptionTransformer):
                transformer = head.transformer
                parameters["head"]["transformer"] = {}

                # Encoder
                if hasattr(transformer, "encoder") and isinstance(transformer.encoder, BEVFormerEncoder):
                    parameters["head"]["transformer"]["encoder"] = extract_transformer_parameters(transformer.encoder)

                # Decoder (MapTR uses MapTRDecoder which extends TransformerLayerSequence)
                if hasattr(transformer, "decoder") and isinstance(
                    transformer.decoder, (DetectionTransformerDecoder, MapTRDecoder)
                ):
                    parameters["head"]["transformer"]["decoder"] = extract_transformer_parameters(transformer.decoder)

                # Reference points
                parameters["head"]["transformer"]["reference_points"] = {
                    "weight": preprocess_linear_weight(transformer.reference_points.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(transformer.reference_points.bias, dtype=ttnn.bfloat16),
                }

                # CAN Bus MLP
                parameters["head"]["transformer"]["can_bus_mlp"] = {
                    "0": {
                        "weight": preprocess_linear_weight(transformer.can_bus_mlp[0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(transformer.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
                    },
                    "2": {
                        "weight": preprocess_linear_weight(transformer.can_bus_mlp[2].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(transformer.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
                    },
                }
                if hasattr(transformer.can_bus_mlp, "norm"):
                    parameters["head"]["transformer"]["can_bus_mlp"]["norm"] = {
                        "weight": preprocess_layernorm_parameter(
                            transformer.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16
                        ),
                        "bias": preprocess_layernorm_parameter(transformer.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16),
                    }

                # Level embeds and camera embeds
                parameters["head"]["transformer"]["level_embeds"] = ttnn.from_torch(
                    transformer.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                    transformer.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )

            # Branches
            parameters["head"]["branches"] = {}
            parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
                head.cls_branches, dtype=ttnn.bfloat16
            )
            parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
                head.reg_branches, dtype=ttnn.bfloat16
            )

        # Process FPN neck
        if hasattr(model, "img_neck") and isinstance(model.img_neck, FPN):
            neck = model.img_neck
            parameters["img_neck"] = {}
            parameters["img_neck"]["lateral_convs"] = {}
            for idx, conv_module in enumerate(neck.lateral_convs):
                parameters["img_neck"]["lateral_convs"][str(idx)] = {
                    "weight": ttnn.from_torch(conv_module.conv.weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(conv_module.conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                }

            parameters["img_neck"]["fpn_convs"] = {}
            for idx, conv_module in enumerate(neck.fpn_convs):
                parameters["img_neck"]["fpn_convs"][str(idx)] = {
                    "weight": ttnn.from_torch(conv_module.conv.weight, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(conv_module.conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                }

        # Process backbone
        if hasattr(model, "img_backbone") and isinstance(model.img_backbone, ResNet):
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
                for block_idx, block in enumerate(layer):
                    prefix = f"layer{layer_idx}_{block_idx}"
                    parameters["img_backbone"][prefix] = {}

                    # conv1, conv2, conv3
                    for conv_name in ["conv1", "conv2", "conv3"]:
                        conv = getattr(block, conv_name)
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["img_backbone"][prefix][conv_name] = {
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
                            parameters["img_backbone"][prefix]["downsample"] = {
                                "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                                "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                            }

    return parameters


def create_maptr_model_parameters(model: MapTR, input_tensor, device=None):
    """Create TTNN parameters from PyTorch MapTR model.

    Args:
        model: The PyTorch MapTR model with loaded weights.
        input_tensor: Input tensor for inferring conv args.
        device: TTNN device.

    Returns:
        Parameters object with preprocessed parameters and conv args.
    """
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {"img_backbone": {}, "img_neck": {}}

    img = input_tensor

    # Handle list input (e.g., [tensor])
    if isinstance(img, list):
        img = img[0]

    # Handle 5D tensor: reshape (B, N, C, H, W) -> (B*N, C, H, W) for conv2d
    if img.dim() == 5:
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)

    parameters.conv_args["img_backbone"] = infer_ttnn_module_args(
        model=model.img_backbone,
        run_model=lambda model: model(img),
        device=None,
    )

    img_feats = model.img_backbone(img)

    # Manually construct Conv2dConfiguration for img_neck since the tracer has name collision issues
    # with ModuleLists that have the same internal structure
    from models.tt_cnn.tt.builder import Conv2dConfiguration

    # Get the FPN module
    fpn = model.img_neck

    # Get input dimensions from backbone output
    # img_feats is a tuple from ResNet, we need the last element
    if isinstance(img_feats, (list, tuple)):
        backbone_output = img_feats[-1]
    else:
        backbone_output = img_feats
    batch_size, _, input_h, input_w = backbone_output.shape

    # Build Conv2dConfiguration for lateral_convs (1x1 conv from backbone to FPN channels)
    lateral_conv = fpn.lateral_convs[0].conv
    lateral_weight, lateral_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
        lateral_conv.weight.detach(), lateral_conv.bias.detach() if lateral_conv.bias is not None else None
    )
    lateral_conv_config = Conv2dConfiguration(
        input_height=input_h,
        input_width=input_w,
        in_channels=lateral_conv.in_channels,
        out_channels=lateral_conv.out_channels,
        batch_size=batch_size,
        kernel_size=lateral_conv.kernel_size,
        weight=lateral_weight,
        stride=lateral_conv.stride,
        padding=lateral_conv.padding,
        groups=lateral_conv.groups,
        dilation=lateral_conv.dilation,
        bias=lateral_bias,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )

    # Output size of lateral conv (1x1 conv preserves spatial dims)
    lateral_out_h = input_h
    lateral_out_w = input_w

    # Build Conv2dConfiguration for fpn_convs (3x3 conv)
    fpn_conv = fpn.fpn_convs[0].conv
    fpn_weight, fpn_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
        fpn_conv.weight.detach(), fpn_conv.bias.detach() if fpn_conv.bias is not None else None
    )
    fpn_conv_config = Conv2dConfiguration(
        input_height=lateral_out_h,
        input_width=lateral_out_w,
        in_channels=fpn_conv.in_channels,
        out_channels=fpn_conv.out_channels,
        batch_size=batch_size,
        kernel_size=fpn_conv.kernel_size,
        weight=fpn_weight,
        stride=fpn_conv.stride,
        padding=fpn_conv.padding,
        groups=fpn_conv.groups,
        dilation=fpn_conv.dilation,
        bias=fpn_bias,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )

    # Create a structure that TtMapTR can use
    from ttnn.dot_access import make_dot_access_dict

    parameters.conv_args["img_neck"] = make_dot_access_dict(
        {
            "lateral_convs": {0: lateral_conv_config},
            "fpn_convs": {0: fpn_conv_config},
        }
    )

    assert parameters is not None

    for key in parameters.conv_args.keys():
        if key == "img_backbone":
            for conv_key in parameters.conv_args[key].keys():
                if hasattr(model.img_backbone, conv_key):
                    parameters.conv_args[key][conv_key].module = getattr(model.img_backbone, conv_key)
        elif key == "img_neck":
            # FPN conv_args are manually constructed Conv2dConfiguration objects
            # No module assignment needed as the configs already contain weights
            pass

    return parameters


def load_maptr_weights(model: MapTR, weights_path: str):
    """Load MapTR weights from checkpoint.

    Args:
        model: MapTR model instance.
        weights_path: Path to checkpoint file.

    Returns:
        Model with loaded weights.
    """
    import os

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return model
