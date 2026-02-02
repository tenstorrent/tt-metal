# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

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
from models.experimental.BEVFormerV2.reference.decoder import (
    DetectionTransformerDecoder,
)
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


def extract_transformer_layers_parameters(transformer_module, parent_model=None):
    """Extract parameters from transformer encoder/decoder layers.

    Args:
        transformer_module: The transformer encoder or decoder module
        parent_model: Optional parent model to check for num_feature_levels mismatch
    """
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
                num_heads = deform_attn.num_heads
                num_levels_checkpoint = deform_attn.num_levels
                num_points = deform_attn.num_points

                if parent_model is not None:
                    transformer_module_parent = None
                    for parent_name, parent_module in parent_model.named_modules():
                        if hasattr(parent_module, "transformer") and hasattr(
                            parent_module.transformer, "num_feature_levels"
                        ):
                            transformer_module_parent = parent_module.transformer
                            break

                    num_feature_levels_used = (
                        transformer_module_parent.num_feature_levels
                        if transformer_module_parent
                        else num_levels_checkpoint
                    )

                    if num_levels_checkpoint > 1 and num_feature_levels_used < num_levels_checkpoint:
                        offsets_per_head = num_levels_checkpoint * num_points * 2
                        offsets_keep = num_points * 2
                        offsets_idx = []
                        for h in range(num_heads):
                            base = h * offsets_per_head
                            offsets_idx.extend(range(base, base + offsets_keep))
                        offsets_idx = torch.tensor(offsets_idx, dtype=torch.long)

                        attn_per_head = num_levels_checkpoint * num_points
                        attn_keep = num_points
                        attn_idx = []
                        for h in range(num_heads):
                            base = h * attn_per_head
                            attn_idx.extend(range(base, base + attn_keep))
                        attn_idx = torch.tensor(attn_idx, dtype=torch.long)

                        sampling_offsets_weight = deform_attn.sampling_offsets.weight.index_select(0, offsets_idx)
                        sampling_offsets_bias = deform_attn.sampling_offsets.bias.index_select(0, offsets_idx)
                        attention_weights_weight = deform_attn.attention_weights.weight.index_select(0, attn_idx)
                        attention_weights_bias = deform_attn.attention_weights.bias.index_select(0, attn_idx)
                    else:
                        sampling_offsets_weight = deform_attn.sampling_offsets.weight
                        sampling_offsets_bias = deform_attn.sampling_offsets.bias
                        attention_weights_weight = deform_attn.attention_weights.weight
                        attention_weights_bias = deform_attn.attention_weights.bias
                elif num_levels_checkpoint > 1:
                    offsets_per_head = num_levels_checkpoint * num_points * 2
                    offsets_keep = num_points * 2
                    offsets_idx = []
                    for h in range(num_heads):
                        base = h * offsets_per_head
                        offsets_idx.extend(range(base, base + offsets_keep))
                    offsets_idx = torch.tensor(offsets_idx, dtype=torch.long)

                    attn_per_head = num_levels_checkpoint * num_points
                    attn_keep = num_points
                    attn_idx = []
                    for h in range(num_heads):
                        base = h * attn_per_head
                        attn_idx.extend(range(base, base + attn_keep))
                    attn_idx = torch.tensor(attn_idx, dtype=torch.long)

                    sampling_offsets_weight = deform_attn.sampling_offsets.weight.index_select(0, offsets_idx)
                    sampling_offsets_bias = deform_attn.sampling_offsets.bias.index_select(0, offsets_idx)
                    attention_weights_weight = deform_attn.attention_weights.weight.index_select(0, attn_idx)
                    attention_weights_bias = deform_attn.attention_weights.bias.index_select(0, attn_idx)
                else:
                    sampling_offsets_weight = deform_attn.sampling_offsets.weight
                    sampling_offsets_bias = deform_attn.sampling_offsets.bias
                    attention_weights_weight = deform_attn.attention_weights.weight
                    attention_weights_bias = deform_attn.attention_weights.bias

                layer_dict["attentions"][f"attn{j}"] = {
                    "sampling_offsets": {
                        "weight": preprocess_linear_weight(sampling_offsets_weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(sampling_offsets_bias, dtype=ttnn.bfloat16),
                    },
                    "attention_weights": {
                        "weight": preprocess_linear_weight(attention_weights_weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(attention_weights_bias, dtype=ttnn.bfloat16),
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


def extract_resnet_parameters(model):
    """Extract ResNet parameters with batch norm folding."""
    parameters = {}

    if isinstance(model.norm1, nn.Identity):
        parameters["conv1"] = {
            "weight": ttnn.from_torch(model.conv1.weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(model.conv1.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }
    else:
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.norm1)
        parameters["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

    for layer_idx in range(1, 5):
        layer = getattr(model, f"layer{layer_idx}")
        for block_idx, block in enumerate(layer):
            prefix = f"layer{layer_idx}_{block_idx}"
            parameters[prefix] = {}

            for conv_name in ["conv1", "conv2", "conv3"]:
                conv = getattr(block, conv_name)
                norm = getattr(block, f"norm{conv_name[-1]}")

                if isinstance(norm, nn.Identity):
                    w = conv.weight
                    b = conv.bias
                else:
                    w, b = fold_batch_norm2d_into_conv2d(conv, norm)

                parameters[prefix][conv_name] = {
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

                    parameters[prefix]["downsample"] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }

    return parameters


def extract_fpn_parameters(model):
    """Extract FPN parameters."""
    parameters = {}
    parameters["lateral_convs"] = []
    for lateral_conv in model.lateral_convs:
        conv = lateral_conv.conv
        lateral_params = {
            "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
            if conv.bias is not None
            else None,
        }
        parameters["lateral_convs"].append(lateral_params)

    parameters["fpn_convs"] = []
    for fpn_conv in model.fpn_convs:
        conv = fpn_conv.conv
        fpn_params = {
            "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
            if conv.bias is not None
            else None,
        }
        parameters["fpn_convs"].append(fpn_params)

    return parameters


def extract_temporal_self_attention_parameters(model):
    """Extract TemporalSelfAttention parameters."""
    return {
        "sampling_offsets": {
            "weight": preprocess_linear_weight(model.sampling_offsets.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.sampling_offsets.bias, dtype=ttnn.bfloat16),
        },
        "attention_weights": {
            "weight": preprocess_linear_weight(model.attention_weights.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.attention_weights.bias, dtype=ttnn.bfloat16),
        },
        "value_proj": {
            "weight": preprocess_linear_weight(model.value_proj.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.value_proj.bias, dtype=ttnn.bfloat16),
        },
        "output_proj": {
            "weight": preprocess_linear_weight(model.output_proj.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.output_proj.bias, dtype=ttnn.bfloat16),
        },
    }


def extract_spatial_cross_attention_parameters(model):
    """Extract SpatialCrossAttention parameters."""
    deform_attn = model.deformable_attention
    return {
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
            "weight": preprocess_linear_weight(model.output_proj.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.output_proj.bias, dtype=ttnn.bfloat16),
        },
    }


def extract_multihead_attention_parameters(model):
    """Extract MultiheadAttention parameters."""
    return {
        "in_proj": {
            "weight": preprocess_linear_weight(model.attn.in_proj_weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.attn.in_proj_bias, dtype=ttnn.bfloat16),
        },
        "out_proj": {
            "weight": preprocess_linear_weight(model.attn.out_proj.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.attn.out_proj.bias, dtype=ttnn.bfloat16),
        },
    }


def extract_custom_ms_deformable_attention_parameters(model):
    """Extract CustomMSDeformableAttention parameters."""
    return {
        "sampling_offsets": {
            "weight": preprocess_linear_weight(model.sampling_offsets.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.sampling_offsets.bias, dtype=ttnn.bfloat16),
        },
        "attention_weights": {
            "weight": preprocess_linear_weight(model.attention_weights.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.attention_weights.bias, dtype=ttnn.bfloat16),
        },
        "value_proj": {
            "weight": preprocess_linear_weight(model.value_proj.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.value_proj.bias, dtype=ttnn.bfloat16),
        },
        "output_proj": {
            "weight": preprocess_linear_weight(model.output_proj.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.output_proj.bias, dtype=ttnn.bfloat16),
        },
    }


def extract_sequential_branch(module_list, dtype):
    """Extract parameters from sequential branches (cls/reg branches)."""
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


def extract_embeddings_to_ttnn(model, names, dtype, device=None):
    result = {}
    for name in names:
        weight = ttnn.from_torch(getattr(model, name).weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        if device is not None:
            weight = ttnn.to_device(weight, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result[name] = {"weight": weight}
    return result


def extract_reg_branches(reg_branches, dtype):
    """Extract regression branch parameters (simplified version without LayerNorm)."""
    branches_params = {}
    for i, branch in enumerate(reg_branches):
        branch_params = {}
        layer_idx = 0
        for layer in branch:
            if isinstance(layer, nn.Linear):
                branch_params[str(layer_idx)] = {
                    "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                    "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
                }
                layer_idx += 1
        branches_params[str(i)] = branch_params
    return branches_params


def prepare_ffn_parameters_for_test(ffn_module, device):
    """Prepare FFN parameters for TTNN test modules (converts to object structure and moves to device)."""

    class Parameters:
        pass

    params = Parameters()
    params.linear1 = Parameters()
    params.linear1.weight = ttnn.to_device(
        preprocess_linear_weight(ffn_module.layers[0][0].weight, dtype=ttnn.bfloat16), device
    )
    params.linear1.bias = (
        ttnn.to_device(preprocess_linear_bias(ffn_module.layers[0][0].bias, dtype=ttnn.bfloat16), device)
        if ffn_module.layers[0][0].bias is not None
        else None
    )

    params.linear2 = Parameters()
    params.linear2.weight = ttnn.to_device(
        preprocess_linear_weight(ffn_module.layers[1].weight, dtype=ttnn.bfloat16), device
    )
    params.linear2.bias = (
        ttnn.to_device(preprocess_linear_bias(ffn_module.layers[1].bias, dtype=ttnn.bfloat16), device)
        if ffn_module.layers[1].bias is not None
        else None
    )

    return params


def _convert_attention_params_dict_to_object(attn_params_dict, device):
    """Convert attention parameter dictionary to object structure expected by TTNN classes."""

    class Parameters:
        pass

    if "in_proj" in attn_params_dict:
        # MultiheadAttention
        attn_params = Parameters()
        attn_params.in_proj = Parameters()
        attn_params.in_proj.weight = ttnn.to_device(attn_params_dict["in_proj"]["weight"], device)
        attn_params.in_proj.bias = ttnn.to_device(attn_params_dict["in_proj"]["bias"], device)
        attn_params.out_proj = Parameters()
        attn_params.out_proj.weight = ttnn.to_device(attn_params_dict["out_proj"]["weight"], device)
        attn_params.out_proj.bias = ttnn.to_device(attn_params_dict["out_proj"]["bias"], device)
    elif "sampling_offsets" in attn_params_dict:
        # CustomMSDeformableAttention
        attn_params = Parameters()
        attn_params.sampling_offsets = Parameters()
        attn_params.sampling_offsets.weight = ttnn.to_device(attn_params_dict["sampling_offsets"]["weight"], device)
        attn_params.sampling_offsets.bias = ttnn.to_device(attn_params_dict["sampling_offsets"]["bias"], device)
        attn_params.attention_weights = Parameters()
        attn_params.attention_weights.weight = ttnn.to_device(attn_params_dict["attention_weights"]["weight"], device)
        attn_params.attention_weights.bias = ttnn.to_device(attn_params_dict["attention_weights"]["bias"], device)
        attn_params.value_proj = Parameters()
        attn_params.value_proj.weight = ttnn.to_device(attn_params_dict["value_proj"]["weight"], device)
        attn_params.value_proj.bias = ttnn.to_device(attn_params_dict["value_proj"]["bias"], device)
        attn_params.output_proj = Parameters()
        attn_params.output_proj.weight = ttnn.to_device(attn_params_dict["output_proj"]["weight"], device)
        attn_params.output_proj.bias = ttnn.to_device(attn_params_dict["output_proj"]["bias"], device)
    else:
        raise ValueError(f"Unknown attention parameter structure: {list(attn_params_dict.keys())}")

    return attn_params


def prepare_decoder_layer_parameters_for_test(decoder_layer, device):
    """Prepare decoder layer parameters for TTNN test modules (converts to object structure and moves to device)."""

    class Parameters:
        pass

    params = Parameters()
    params.attentions = {}
    params.ffn = {}
    params.norms = {}

    for j, attn in enumerate(decoder_layer.attentions):
        if isinstance(attn, MultiheadAttention):
            attn_params_dict = extract_multihead_attention_parameters(attn)
        elif isinstance(attn, CustomMSDeformableAttention):
            attn_params_dict = extract_custom_ms_deformable_attention_parameters(attn)
        else:
            raise ValueError(f"Unsupported attention type: {type(attn)}")
        params.attentions[f"attn{j}"] = _convert_attention_params_dict_to_object(attn_params_dict, device)

    for k, ffn in enumerate(decoder_layer.ffns):
        params.ffn[f"ffn{k}"] = prepare_ffn_parameters_for_test(ffn, device)

    for n, norm in enumerate(decoder_layer.norms):
        norm_params = Parameters()
        norm_params.weight = ttnn.to_device(preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16), device)
        norm_params.bias = ttnn.to_device(preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16), device)
        params.norms[f"norm{n}"] = norm_params

    return params


def custom_preprocessor(model, name):
    """Main custom preprocessor that handles both full models and standalone modules."""
    parameters = {}

    if isinstance(model, ResNet):
        parameters.update(extract_resnet_parameters(model))
        return parameters

    if isinstance(model, FPN):
        parameters.update(extract_fpn_parameters(model))
        return parameters

    if isinstance(model, BEVFormerEncoder):
        parameters.update(extract_transformer_layers_parameters(model))
        return parameters

    if isinstance(model, DetectionTransformerDecoder):
        parameters.update(extract_transformer_layers_parameters(model))
        return parameters

    if isinstance(model, TemporalSelfAttention):
        parameters["temporal_self_attention"] = extract_temporal_self_attention_parameters(model)
        return parameters

    if isinstance(model, SpatialCrossAttention):
        parameters["spatial_cross_attention"] = extract_spatial_cross_attention_parameters(model)
        return parameters

    if isinstance(model, MultiheadAttention):
        parameters["multihead_attention"] = extract_multihead_attention_parameters(model)
        return parameters

    if isinstance(model, CustomMSDeformableAttention):
        parameters["custom_ms_deformable_attention"] = extract_custom_ms_deformable_attention_parameters(model)
        return parameters

    if isinstance(model, PerceptionTransformerV2):
        parameters = {}

        if isinstance(model.encoder, BEVFormerEncoder):
            parameters["encoder"] = extract_transformer_layers_parameters(model.encoder, parent_model=model)

        if isinstance(model.decoder, DetectionTransformerDecoder):
            parameters["decoder"] = extract_transformer_layers_parameters(model.decoder, parent_model=model)

        parameters["reference_points"] = {
            "weight": preprocess_linear_weight(model.reference_points.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.reference_points.bias, dtype=ttnn.bfloat16),
        }

        parameters["map_reference_points"] = {
            "weight": preprocess_linear_weight(model.reference_points.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.reference_points.bias, dtype=ttnn.bfloat16),
        }

        parameters["level_embeds"] = ttnn.from_torch(model.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if hasattr(model, "cams_embeds"):
            parameters["cams_embeds"] = ttnn.from_torch(model.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            parameters["cams_embeds"] = ttnn.from_torch(
                torch.zeros(6, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

        return parameters

    if isinstance(model, BEVFormerHead):
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

        if isinstance(model.transformer, PerceptionTransformerV2):
            parameters["head"]["transformer"] = {}
            if isinstance(model.transformer.encoder, BEVFormerEncoder):
                parameters["head"]["transformer"]["encoder"] = extract_transformer_layers_parameters(
                    model.transformer.encoder, parent_model=model
                )

            if isinstance(model.transformer.decoder, DetectionTransformerDecoder):
                parameters["head"]["transformer"]["decoder"] = extract_transformer_layers_parameters(
                    model.transformer.decoder, parent_model=model
                )

            parameters["head"]["transformer"]["reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.reference_points.bias, dtype=ttnn.bfloat16),
            }

            parameters["head"]["transformer"]["map_reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.reference_points.bias, dtype=ttnn.bfloat16),
            }

            if hasattr(model.transformer, "level_embeds"):
                parameters["head"]["transformer"]["level_embeds"] = ttnn.from_torch(
                    model.transformer.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            if hasattr(model.transformer, "cams_embeds"):
                parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                    model.transformer.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            else:
                parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                    torch.zeros(6, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )

            parameters["head"]["transformer"]["decoder"]["reg_branches"] = extract_reg_branches(
                model.reg_branches, dtype=ttnn.bfloat16
            )

        embedding_layers = ["bev_embedding", "query_embedding"]
        parameters["head"].update(extract_embeddings_to_ttnn(model, embedding_layers, dtype=ttnn.bfloat16, device=None))
        parameters["head"]["branches"] = {}

        parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
            model.cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
            model.reg_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["cls_branches_torch"] = model.cls_branches
        parameters["head"]["reg_branches_torch"] = model.reg_branches

        return parameters

    if isinstance(model, (BEVFormerV2, DetectionTransformerDecoder)):
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
                    parameters["head"]["transformer"]["encoder"] = extract_transformer_layers_parameters(
                        head.transformer.encoder, parent_model=model
                    )

                if isinstance(head.transformer.decoder, DetectionTransformerDecoder):
                    parameters["head"]["transformer"]["decoder"] = extract_transformer_layers_parameters(
                        head.transformer.decoder, parent_model=model
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
            parameters["head"].update(
                extract_embeddings_to_ttnn(head, embedding_layers, dtype=ttnn.bfloat16, device=None)
            )
            parameters["head"]["branches"] = {}

            parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
                head.cls_branches, dtype=ttnn.bfloat16
            )
            parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
                head.reg_branches, dtype=ttnn.bfloat16
            )

        if isinstance(model.img_neck, FPN):
            parameters["img_neck"] = extract_fpn_parameters(model.img_neck)

        if isinstance(model.img_backbone, ResNet):
            parameters["img_backbone"] = extract_resnet_parameters(model.img_backbone)

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
        img = img[0]
        if isinstance(img, torch.Tensor):
            img = img.detach().clone()
        else:
            img = torch.tensor(img)
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

        if device is not None:
            embedding_names = ["bev_embedding", "query_embedding"]
            for emb_name in embedding_names:
                if hasattr(parameters.head, emb_name):
                    emb_obj = getattr(parameters.head, emb_name)
                    if isinstance(emb_obj, dict) and "weight" in emb_obj:
                        emb_weight = emb_obj["weight"]
                        if isinstance(emb_weight, ttnn.Tensor):
                            emb_obj["weight"] = ttnn.to_device(
                                emb_weight, device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                            )
                    elif hasattr(emb_obj, "weight"):
                        emb_weight = getattr(emb_obj, "weight")
                        if isinstance(emb_weight, ttnn.Tensor):
                            setattr(
                                emb_obj,
                                "weight",
                                ttnn.to_device(emb_weight, device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                            )

    return parameters
