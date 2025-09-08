# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def fold_batch_norm2d_into_conv2d(conv, bn, eps=1e-05):
    bn_weight = bn.weight.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = bn.bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = bn.running_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = bn.running_var.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = conv.weight
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    return weight, bias


def custom_preprocessor(model, name, mesh_mapper=None):
    parameters = {}
    if hasattr(model, "conv") and hasattr(model, "bn"):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        parameters["conv"]["bias"] = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1), dtype=ttnn.float32, mesh_mapper=mesh_mapper
        )

    elif hasattr(model, "conv_dw") and hasattr(model, "conv_pw") and hasattr(model, "bn"):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv_pw, model.bn)
        parameters["conv_pw"] = {}
        parameters["conv_pw"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        parameters["conv_pw"]["bias"] = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1), dtype=ttnn.float32, mesh_mapper=mesh_mapper
        )

        parameters["conv_dw"] = {}
        parameters["conv_dw"]["weight"] = ttnn.from_torch(
            model.conv_dw.weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper
        )
        if model.conv_dw.bias is not None:
            dw_bias = model.conv_dw.bias.reshape(1, 1, 1, -1)
            parameters["conv_dw"]["bias"] = ttnn.from_torch(dw_bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    elif isinstance(model, torch.nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        if model.bias is not None:
            bias = model.bias.reshape(1, 1, 1, -1)
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor
