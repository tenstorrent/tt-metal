# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.maptr.reference.resnet import ResNet
from models.experimental.maptr.reference.fpn import FPN
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, ResNet):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"] = {}
        parameters["res_model"]["conv1"] = {}
        parameters["res_model"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer1[0].conv1, model.layer1[0].bn1)
        parameters["res_model"]["layer1_0"] = {}
        parameters["res_model"]["layer1_0"]["conv1"] = {}
        parameters["res_model"]["layer1_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer1[0].conv2, model.layer1[0].bn2)
        parameters["res_model"]["layer1_0"]["conv2"] = {}
        parameters["res_model"]["layer1_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer1[1].conv1, model.layer1[1].bn1)
        parameters["res_model"]["layer1_1"] = {}
        parameters["res_model"]["layer1_1"]["conv1"] = {}
        parameters["res_model"]["layer1_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer1[1].conv2, model.layer1[1].bn2)
        parameters["res_model"]["layer1_1"]["conv2"] = {}
        parameters["res_model"]["layer1_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer2[0].conv1, model.layer2[0].bn1)
        parameters["res_model"]["layer2_0"] = {}
        parameters["res_model"]["layer2_0"]["conv1"] = {}
        parameters["res_model"]["layer2_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer2[0].conv2, model.layer2[0].bn2)
        parameters["res_model"]["layer2_0"]["conv2"] = {}
        parameters["res_model"]["layer2_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer2[1].conv1, model.layer2[1].bn1)
        parameters["res_model"]["layer2_1"] = {}
        parameters["res_model"]["layer2_1"]["conv1"] = {}
        parameters["res_model"]["layer2_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer2[1].conv2, model.layer2[1].bn2)
        parameters["res_model"]["layer2_1"]["conv2"] = {}
        parameters["res_model"]["layer2_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        if hasattr(model.layer2[0], "downsample") and model.layer2[0].downsample is not None:
            downsample = model.layer2[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["res_model"]["layer2_0"]["downsample"] = {}
                parameters["res_model"]["layer2_0"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.float32
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["res_model"]["layer2_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer3[0].conv1, model.layer3[0].bn1)
        parameters["res_model"]["layer3_0"] = {}
        parameters["res_model"]["layer3_0"]["conv1"] = {}
        parameters["res_model"]["layer3_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer3[0].conv2, model.layer3[0].bn2)
        parameters["res_model"]["layer3_0"]["conv2"] = {}
        parameters["res_model"]["layer3_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer3[1].conv1, model.layer3[1].bn1)
        parameters["res_model"]["layer3_1"] = {}
        parameters["res_model"]["layer3_1"]["conv1"] = {}
        parameters["res_model"]["layer3_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer3[1].conv2, model.layer3[1].bn2)
        parameters["res_model"]["layer3_1"]["conv2"] = {}
        parameters["res_model"]["layer3_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        if hasattr(model.layer3[0], "downsample") and model.layer3[0].downsample is not None:
            downsample = model.layer3[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["res_model"]["layer3_0"]["downsample"] = {}
                parameters["res_model"]["layer3_0"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.float32
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["res_model"]["layer3_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer4[0].conv1, model.layer4[0].bn1)
        parameters["res_model"]["layer4_0"] = {}
        parameters["res_model"]["layer4_0"]["conv1"] = {}
        parameters["res_model"]["layer4_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer4[0].conv2, model.layer4[0].bn2)
        parameters["res_model"]["layer4_0"]["conv2"] = {}
        parameters["res_model"]["layer4_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer4[1].conv1, model.layer4[1].bn1)
        parameters["res_model"]["layer4_1"] = {}
        parameters["res_model"]["layer4_1"]["conv1"] = {}
        parameters["res_model"]["layer4_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer4[1].conv2, model.layer4[1].bn2)
        parameters["res_model"]["layer4_1"]["conv2"] = {}
        parameters["res_model"]["layer4_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        if hasattr(model.layer4[0], "downsample") and model.layer4[0].downsample is not None:
            downsample = model.layer4[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["res_model"]["layer4_0"]["downsample"] = {}
                parameters["res_model"]["layer4_0"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.float32
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["res_model"]["layer4_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
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

    return parameters


def create_maptr_model_parameters(model: ResNet, input_tensor, device=None):
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
