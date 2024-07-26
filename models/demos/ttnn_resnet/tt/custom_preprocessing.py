# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    fold_batch_norm2d_into_conv2d,
    convert_torch_model_to_ttnn_model,
)
from models.utility_functions import pad_and_fold_conv_filters_for_unity_stride


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    if isinstance(model, torchvision.models.resnet.Bottleneck):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)
        parameters["conv1"] = {}
        parameters["conv2"] = {}
        parameters["conv3"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(conv1_weight, mesh_mapper=mesh_mapper)
        parameters["conv2"]["weight"] = ttnn.from_torch(conv2_weight, mesh_mapper=mesh_mapper)
        parameters["conv3"]["weight"] = ttnn.from_torch(conv3_weight, mesh_mapper=mesh_mapper)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(conv1_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        parameters["conv2"]["bias"] = ttnn.from_torch(torch.reshape(conv2_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        parameters["conv3"]["bias"] = ttnn.from_torch(torch.reshape(conv3_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        if model.downsample is not None:
            downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(model.downsample[0], model.downsample[1])
            parameters["downsample"] = {}
            parameters["downsample"]["weight"] = ttnn.from_torch(downsample_weight, mesh_mapper=mesh_mapper)
            parameters["downsample"]["bias"] = ttnn.from_torch(
                torch.reshape(downsample_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
            )
    elif isinstance(model, torchvision.models.resnet.ResNet):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv1_weight = pad_and_fold_conv_filters_for_unity_stride(conv1_weight, 2, 2)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(conv1_weight, mesh_mapper=mesh_mapper)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(conv1_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        named_parameters = tuple((name, parameter) for name, parameter in model.named_parameters() if "." not in name)
        for child_name, child in tuple(model.named_children()) + named_parameters:
            if child_name in {"conv1", "bn1"}:
                continue
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=name,
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
