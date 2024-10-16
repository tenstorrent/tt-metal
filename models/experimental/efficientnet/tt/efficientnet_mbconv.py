# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math

from dataclasses import dataclass
from typing import List, Callable, Optional

from models.experimental.efficientnet.tt.efficientnet_conv import TtEfficientnetConv2dNormActivation
from models.experimental.efficientnet.tt.efficientnet_squeeze_excitation import (
    TtEfficientnetSqueezeExcitation,
)


def _make_divisible(v: float, divisor: int, min_value=None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., torch.nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        do_adjust_depth: bool = True,
        do_adjust_input_channels: bool = True,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., torch.nn.Module]] = None,
    ):
        if do_adjust_input_channels:
            input_channels = self.adjust_channels(input_channels, width_mult)

        out_channels = self.adjust_channels(out_channels, width_mult)

        if do_adjust_depth:
            num_layers = self.adjust_depth(num_layers, depth_mult)

        if block is None:
            block = TtEfficientnetMbConv

        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class TtEfficientnetMbConv(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.SiLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        state_dict,
        base_address,
        device,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer_eps: float = 1e-05,
        norm_layer_momentum: float = 0.1,
        is_lite: bool = False,
    ):
        super().__init__()

        # Efficientnet Lite has hardcoded eps and momentum
        if is_lite:
            norm_layer_momentum = 0.01
            norm_layer_eps = 1e-3

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[torch.nn.Module] = []

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        if expanded_channels != cnf.input_channels:
            layers.append(
                TtEfficientnetConv2dNormActivation(
                    state_dict=state_dict,
                    conv_base_address=(
                        f"{base_address}._expand_conv" if is_lite else f"{base_address}.block.{len(layers)}.0"
                    ),
                    bn_base_address=f"{base_address}._bn0" if is_lite else f"{base_address}.block.{len(layers)}.1",
                    device=device,
                    in_channels=cnf.input_channels,
                    out_channels=expanded_channels,
                    kernel_size=1,
                    norm_layer_eps=norm_layer_eps,
                    norm_layer_momentum=norm_layer_momentum,
                    is_lite=is_lite,
                )
            )

        # depthwise
        layers.append(
            TtEfficientnetConv2dNormActivation(
                state_dict=state_dict,
                conv_base_address=(
                    f"{base_address}._depthwise_conv" if is_lite else f"{base_address}.block.{len(layers)}.0"
                ),
                bn_base_address=f"{base_address}._bn1" if is_lite else f"{base_address}.block.{len(layers)}.1",
                device=device,
                in_channels=expanded_channels,
                out_channels=expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer_eps=norm_layer_eps,
                norm_layer_momentum=norm_layer_momentum,
                is_lite=is_lite,
            )
        )

        # Efficientnet Lite does not have SqueezeExcitation layer
        if not is_lite:
            # squeeze and excitation
            squeeze_channels = max(1, cnf.input_channels // 4)

            layers.append(
                TtEfficientnetSqueezeExcitation(
                    state_dict=state_dict,
                    base_address=f"{base_address}.block.{len(layers)}",
                    device=device,
                    input_channels=expanded_channels,
                    squeeze_channels=squeeze_channels,
                )
            )

        # project
        layers.append(
            TtEfficientnetConv2dNormActivation(
                state_dict=state_dict,
                conv_base_address=(
                    f"{base_address}._project_conv" if is_lite else f"{base_address}.block.{len(layers)}.0"
                ),
                bn_base_address=f"{base_address}._bn2" if is_lite else f"{base_address}.block.{len(layers)}.1",
                device=device,
                in_channels=expanded_channels,
                out_channels=cnf.out_channels,
                kernel_size=1,
                norm_layer_eps=norm_layer_eps,
                norm_layer_momentum=norm_layer_momentum,
                activation_layer=False,
                is_lite=is_lite,
            )
        )

        self.block = torch.nn.Sequential(*layers)
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, x):
        result = self.block(x)

        if self.use_res_connect:
            # result = self.stochastic_depth(result)
            result = ttnn.add(result, x)

        return result
