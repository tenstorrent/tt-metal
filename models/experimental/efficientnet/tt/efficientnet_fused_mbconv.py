# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import List, Optional, Callable

from models.experimental.efficientnet.tt.efficientnet_conv import TtEfficientnetConv2dNormActivation
from models.experimental.efficientnet.tt.efficientnet_mbconv import _MBConvConfig


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., torch.nn.Module]] = None,
    ):
        if block is None:
            block = TtEfficientnetFusedMBConv

        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class TtEfficientnetFusedMBConv(torch.nn.Module):
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer_eps: float = 1e-05,
        norm_layer_momentum: float = 0.1,
        is_lite: bool = False,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[torch.nn.Module] = []
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                TtEfficientnetConv2dNormActivation(
                    state_dict=state_dict,
                    conv_base_address=f"{base_address}.block.{len(layers)}.0",
                    bn_base_address=f"{base_address}.block.{len(layers)}.1",
                    device=device,
                    in_channels=cnf.input_channels,
                    out_channels=expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer_eps=norm_layer_eps,
                    norm_layer_momentum=norm_layer_momentum,
                    activation_layer=True,
                )
            )

            # project
            layers.append(
                TtEfficientnetConv2dNormActivation(
                    state_dict=state_dict,
                    conv_base_address=f"{base_address}.block.{len(layers)}.0",
                    bn_base_address=f"{base_address}.block.{len(layers)}.1",
                    device=device,
                    in_channels=expanded_channels,
                    out_channels=cnf.out_channels,
                    kernel_size=1,
                    norm_layer_eps=norm_layer_eps,
                    norm_layer_momentum=norm_layer_momentum,
                    activation_layer=False,
                )
            )
        else:
            layers.append(
                TtEfficientnetConv2dNormActivation(
                    state_dict=state_dict,
                    conv_base_address=f"{base_address}.block.{len(layers)}.0",
                    bn_base_address=f"{base_address}.block.{len(layers)}.1",
                    device=device,
                    in_channels=cnf.input_channels,
                    out_channels=cnf.out_channels,
                    kernel_size=cnf.kernel,
                    norm_layer_eps=norm_layer_eps,
                    norm_layer_momentum=norm_layer_momentum,
                    activation_layer=True,
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
