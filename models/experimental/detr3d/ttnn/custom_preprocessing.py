# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d

from models.experimental.detr3d.reference.model_3detr import GenericMLP
from models.experimental.detr3d.reference.model_3detr import MaskedTransformerEncoder, TransformerEncoderLayer
from models.experimental.detr3d.reference.pointnet2_modules import SharedMLP


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def fold_batch_norm1d_into_conv1d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm1d must have track_running_stats=True to be folded into Conv1d")

    weight = conv.weight  # Shape: [out_channels, in_channels, kernel_size]
    bias = conv.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias

    # For 1D: scale factor applied per output channel
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None]

    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    # For 1D convolutions, bias shape should be [1, 1, -1] instead of [1, 1, 1, -1]
    bias = bias.reshape(1, 1, 1, -1)

    return weight, bias


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    weight_dtype = ttnn.bfloat16
    if isinstance(model, GenericMLP):
        mlp_layers = []
        for child_name, child in model.layers.named_children():
            mlp_layers.append(child)
        parameters["layers"] = {}
        for layer_num, layer in enumerate(mlp_layers):
            parameters["layers"][layer_num] = {}
            if isinstance(layer, torch.nn.Conv1d):
                if (layer_num + 1) < len(mlp_layers):
                    next_layer = mlp_layers[layer_num + 1]
                    if isinstance(next_layer, torch.nn.BatchNorm1d):
                        weight, bias = fold_batch_norm1d_into_conv1d(layer, next_layer)
                        parameters["layers"][layer_num]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
                        parameters["layers"][layer_num]["bias"] = ttnn.from_torch(bias, mesh_mapper=mesh_mapper)
                        continue
                weight = layer.weight
                if layer.bias is not None:
                    bias = layer.bias
                    if bias.dim() < 4:
                        bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    parameters["layers"][layer_num]["bias"] = ttnn.from_torch(bias, mesh_mapper=mesh_mapper)
                parameters["layers"][layer_num]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
    elif isinstance(model, SharedMLP):
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer0.conv, model.layer0.bn.bn)
        parameters["layer0"] = {}
        parameters["layer0"]["conv"] = {}
        parameters["layer0"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["layer0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer1.conv, model.layer1.bn.bn)
        parameters["layer1"] = {}
        parameters["layer1"]["conv"] = {}
        parameters["layer1"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["layer1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer2.conv, model.layer2.bn.bn)
        parameters["layer2"] = {}
        parameters["layer2"]["conv"] = {}
        parameters["layer2"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["layer2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    elif isinstance(model, torch.nn.MultiheadAttention):
        # Handle QKV weights for self-attention
        if hasattr(model, "in_proj_weight"):
            # Split combined QKV weight into separate Q, K, V
            qkv_weight = model.in_proj_weight
            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)

            parameters["q_weight"] = ttnn.from_torch(q_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
            parameters["k_weight"] = ttnn.from_torch(k_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
            parameters["v_weight"] = ttnn.from_torch(v_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)

        if hasattr(model, "in_proj_bias") and model.in_proj_bias is not None:
            qkv_bias = model.in_proj_bias
            q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
            parameters["q_bias"] = ttnn.from_torch(q_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
            parameters["k_bias"] = ttnn.from_torch(k_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
            parameters["v_bias"] = ttnn.from_torch(v_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)

        if hasattr(model, "out_proj"):
            parameters["out_weight"] = ttnn.from_torch(
                model.out_proj.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
            )
            parameters["out_bias"] = None
            if model.out_proj.bias is not None:
                parameters["out_bias"] = ttnn.from_torch(
                    model.out_proj.bias.reshape(1, -1),
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                )

    # Preprocess feedforward parameters
    elif isinstance(model, torch.nn.Linear):
        parameters["weight"] = ttnn.from_torch(model.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
        if hasattr(model, "bias") and model.bias is not None:
            parameters["bias"] = ttnn.from_torch(model.bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)

    # Preprocess layer normalization parameters
    elif isinstance(model, torch.nn.LayerNorm):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
        if hasattr(model, "bias") and model.bias is not None:
            parameters["bias"] = ttnn.from_torch(model.bias, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)

    elif isinstance(
        model,
        (
            MaskedTransformerEncoder,
            TransformerEncoderLayer,
        ),
    ):
        # Let the sub-modules handle their own preprocessing
        for child_name, child in model.named_children():
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=f"{name}.{child_name}",
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
