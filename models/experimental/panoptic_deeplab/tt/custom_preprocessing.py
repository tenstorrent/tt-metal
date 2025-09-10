# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision
from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d

import ttnn
from models.utility_functions import pad_and_fold_conv_filters_for_unity_stride
from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem

from models.experimental.panoptic_deeplab.reference.head import (
    HeadModel,
)
from models.experimental.panoptic_deeplab.reference.res_block import (
    ResModel,
)
from models.experimental.panoptic_deeplab.reference.aspp import (
    PanopticDeeplabASPPModel,
)
from models.experimental.panoptic_deeplab.reference.decoder import (
    DecoderModel,
)


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
    elif isinstance(model, DeepLabStem):
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

    elif isinstance(model, HeadModel):
        for name, module in model.named_children():
            if hasattr(module, "__getitem__"):
                if len(module) > 1 and hasattr(module[0], "weight") and hasattr(module[1], "weight"):
                    # Assume Conv + BN, fold BN into Conv
                    weight, bias = fold_batch_norm2d_into_conv2d(module[0], module[1])
                elif hasattr(module[0], "weight"):
                    # Just a Conv, no BN
                    weight = module[0].weight.clone().detach().contiguous()
                    bias = (
                        module[0].bias.clone().detach().contiguous()
                        if module[0].bias is not None
                        else torch.zeros(module[0].out_channels)
                    )
                else:
                    continue
            elif hasattr(module, "weight"):
                # Single Conv2d
                weight = module.weight.clone().detach().contiguous()
                bias = (
                    module.bias.clone().detach().contiguous()
                    if module.bias is not None
                    else torch.zeros(module.out_channels)
                )
            else:
                continue

            parameters[name] = {}
            parameters[name]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
            parameters[name]["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)

    elif isinstance(model, PanopticDeeplabASPPModel):
        for name, module in model.named_children():
            # For each submodule (e.g., ASPP_0_Conv, ASPP_1_Depthwise, etc.)
            if hasattr(module, "__getitem__"):
                # If it's a Sequential or similar
                if len(module) > 1 and hasattr(module[0], "weight") and hasattr(module[1], "weight"):
                    # Assume Conv + BN, fold BN into Conv
                    weight, bias = fold_batch_norm2d_into_conv2d(module[0], module[1])
                elif hasattr(module[0], "weight"):
                    # Just a Conv, no BN
                    weight = module[0].weight.clone().detach().contiguous()
                    bias = (
                        module[0].bias.clone().detach().contiguous()
                        if module[0].bias is not None
                        else torch.zeros(module[0].out_channels)
                    )
                else:
                    continue
            elif hasattr(module, "weight"):
                # Single Conv2d
                weight = module.weight.clone().detach().contiguous()
                bias = (
                    module.bias.clone().detach().contiguous()
                    if module.bias is not None
                    else torch.zeros(module.out_channels)
                )
            else:
                continue

            parameters[name] = {}
            parameters[name]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
            parameters[name]["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)

    elif isinstance(model, ResModel):
        for name, module in model.named_children():
            if hasattr(module, "__getitem__"):
                # If it's a Sequential or similar
                if len(module) > 1 and hasattr(module[0], "weight") and hasattr(module[1], "weight"):
                    # Assume Conv + BN, fold BN into Conv
                    weight, bias = fold_batch_norm2d_into_conv2d(module[0], module[1])
                elif hasattr(module[0], "weight"):
                    # Just a Conv, no BN
                    weight = module[0].weight.clone().detach().contiguous()
                    bias = (
                        module[0].bias.clone().detach().contiguous()
                        if module[0].bias is not None
                        else torch.zeros(module[0].out_channels)
                    )
                else:
                    continue
            elif hasattr(module, "weight"):
                # Single Conv2d
                weight = module.weight.clone().detach().contiguous()
                bias = (
                    module.bias.clone().detach().contiguous()
                    if module.bias is not None
                    else torch.zeros(module.out_channels)
                )
            else:
                continue

            parameters[name] = {}
            parameters[name]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
            parameters[name]["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)

    elif isinstance(model, DecoderModel):
        parameters = {}
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
