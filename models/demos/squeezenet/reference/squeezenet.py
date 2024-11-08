# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn


def Fire(
    inplanes: int,
    squeeze_planes: int,
    expand1x1_planes: int,
    expand3x3_planes: int,
    input,
    layer_idx: int,
    state_dict=None,
):
    squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    squeeze.weight = nn.Parameter(state_dict[f"features.{layer_idx}.squeeze.weight"])
    squeeze.bias = nn.Parameter(state_dict[f"features.{layer_idx}.squeeze.bias"])
    x = squeeze(input)
    squeeze_activation = nn.ReLU(inplace=True)
    x = squeeze_activation(x)
    expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
    expand1x1.weight = nn.Parameter(state_dict[f"features.{layer_idx}.expand1x1.weight"])
    expand1x1.bias = nn.Parameter(state_dict[f"features.{layer_idx}.expand1x1.bias"])
    expand1x1_output = expand1x1(x)
    expand1x1_activation = nn.ReLU(inplace=True)
    expand1x1_output = expand1x1_activation(expand1x1_output)
    expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
    expand3x3.weight = nn.Parameter(state_dict[f"features.{layer_idx}.expand3x3.weight"])
    expand3x3.bias = nn.Parameter(state_dict[f"features.{layer_idx}.expand3x3.bias"])
    expand3x3_output = expand3x3(x)
    expand3x3_activation = nn.ReLU(inplace=True)
    expand3x3_output = expand3x3_activation(expand3x3_output)
    return torch.cat([expand1x1_output, expand3x3_output], 1)


def squeezenet(state_dict, input):
    first_conv = nn.Conv2d(3, 96, kernel_size=7, stride=2)
    first_conv.weight = nn.Parameter(state_dict["features.0.weight"])
    first_conv.bias = nn.Parameter(state_dict["features.0.bias"])
    relu = nn.ReLU(inplace=True)
    pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
    final_conv = nn.Conv2d(512, 1000, kernel_size=1)
    final_conv.weight = nn.Parameter(state_dict["classifier.1.weight"])
    final_conv.bias = nn.Parameter(state_dict["classifier.1.bias"])
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    x = first_conv(input)
    x = relu(x)
    x = pool(x)
    x = Fire(96, 16, 64, 64, input=x, state_dict=state_dict, layer_idx=3)
    x = Fire(128, 16, 64, 64, input=x, state_dict=state_dict, layer_idx=4)
    x = Fire(128, 32, 128, 128, input=x, state_dict=state_dict, layer_idx=5)
    x = pool(x)
    x = Fire(256, 32, 128, 128, input=x, state_dict=state_dict, layer_idx=7)
    x = Fire(256, 48, 192, 192, input=x, state_dict=state_dict, layer_idx=8)
    x = Fire(384, 48, 192, 192, input=x, state_dict=state_dict, layer_idx=9)
    x = Fire(384, 64, 256, 256, input=x, state_dict=state_dict, layer_idx=10)
    x = pool(x)
    x = Fire(512, 64, 256, 256, input=x, state_dict=state_dict, layer_idx=12)
    x = final_conv(x)
    x = relu(x)
    x = avgpool(x)
    return torch.flatten(x, 1)
