# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import torch
import torch.nn as nn


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    state_dict=None,
    base_address=None,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

    conv.weight = nn.Parameter(state_dict[f"{base_address}.weight"])
    return conv


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, state_dict=None, base_address=None) -> nn.Conv2d:
    """1x1 convolution"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    conv.weight = nn.Parameter(state_dict[f"{base_address}.weight"])
    return conv


def fold_bn_to_conv_weights_bias(conv_weight, bn: torch.nn.BatchNorm2d) -> Tuple[nn.Parameter]:
    # Note: this function is not used, however I am keeping it for reference
    epsilon = bn.eps  # Crucially important to use batchnorm's eps

    bn_weight = bn.weight.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = bn.running_var.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = conv_weight
    weight = (conv_weight / torch.sqrt(bn_running_var + epsilon)) * bn_weight

    bn_running_mean = bn.running_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = bn.bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + epsilon)) + bn_bias

    bias = bias.squeeze(-1).squeeze(-1).squeeze(-1)

    return (weight, bias)


def fold_bn_to_conv(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> Tuple[nn.Parameter]:
    (weight, bias) = fold_bn_to_conv_weights_bias(conv.weight, bn)

    return (nn.Parameter(weight), nn.Parameter(bias))
