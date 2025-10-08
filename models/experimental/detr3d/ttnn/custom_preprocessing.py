# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d

from models.experimental.detr3d.reference.detr3d_model import SharedMLP
from models.experimental.detr3d.reference.detr3d_model import MaskedTransformerEncoder, TransformerEncoderLayer


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    weight_dtype = ttnn.bfloat16
    if isinstance(model, SharedMLP):
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
