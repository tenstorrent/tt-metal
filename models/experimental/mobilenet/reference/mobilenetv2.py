# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# coding=utf-8
# Copyright 2022 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch MobileNetV2 model."""

from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from models.utility_functions import (
    is_conv_supported_on_device,
    run_conv_on_device_wrapper,
)

ACT_FN = nn.ReLU6()


def assign_weight_conv(conv: nn.Conv2d, state_dict, key_w: str):
    conv.weight = nn.Parameter(state_dict[f"{key_w}.weight"])
    if conv.bias is not None:
        conv.bias = nn.Parameter(state_dict[f"{key_w}.bias"])


def assign_weight_batchnorm(norm: nn.BatchNorm2d, state_dict, key_w: str):
    norm.weight = nn.Parameter(state_dict[f"{key_w}.weight"])
    norm.bias = nn.Parameter(state_dict[f"{key_w}.bias"])
    norm.running_mean = nn.Parameter(state_dict[f"{key_w}.running_mean"])
    norm.running_var = nn.Parameter(state_dict[f"{key_w}.running_var"])
    norm.num_batches_tracked = nn.Parameter(state_dict[f"{key_w}.num_batches_tracked"], requires_grad=False)
    norm.eval()


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def apply_depth_multiplier(config, channels: int) -> int:
    return make_divisible(
        int(round(channels * config.depth_multiplier)),
        config.depth_divisible_by,
        config.min_depth,
    )


def apply_tf_padding(features: torch.Tensor, stride, kernel_size, dilation) -> torch.Tensor:
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    in_height = int(features.shape[-2])
    in_width = int(features.shape[-1])
    stride_height, stride_width = stride, stride
    kernel_height, kernel_width = kernel_size, kernel_size
    dilation_height, dilation_width = dilation, dilation

    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    padding = (
        pad_left * dilation_width,
        pad_right * dilation_width,
        pad_top * dilation_height,
        pad_bottom * dilation_height,
    )
    return nn.functional.pad(features, padding, "constant", 0.0)


class MobileNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
        layer_norm_eps: Optional[float] = None,
        state_dict=None,
        base_address="",
        device=None,
        host=None,
        disable_conv_on_tt_device=True,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.host = host
        self.conv_stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2) * dilation

        self.conv_params = [
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            stride,
            stride,
            padding,
            padding,
            dilation,
            groups,
        ]
        if not disable_conv_on_tt_device and is_conv_supported_on_device(self.conv_params):
            conv_weight = state_dict[f"{base_address}.convolution.weight"]
            conv_bias = None
            if bias:
                conv_bias = state_dict[f"{base_address}.convolution.bias"]
            self.convolution = run_conv_on_device_wrapper(
                conv_weight.reshape(-1).tolist(),
                self.conv_params,
                self.device,
                conv_bias,
            )
        else:
            self.convolution = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode="zeros",
            )
            assign_weight_conv(self.convolution, state_dict, f"{base_address}.convolution")

        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=config.layer_norm_eps if layer_norm_eps is None else layer_norm_eps,
                momentum=0.997,
                affine=True,
                track_running_stats=True,
            )
            assign_weight_batchnorm(self.normalization, state_dict, f"{base_address}.normalization")
        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT_FN
            elif isinstance(config.hidden_act, str):
                self.activation = ACT_FN
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.conv_stride, self.kernel_size, self.dilation)
        features = self.convolution(features)
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features


class MobileNetV2InvertedResidual(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = 1,
        state_dict=None,
        base_address="",
        device=None,
        host=None,
        disable_conv_on_tt_device=True,
    ) -> None:
        super().__init__()
        expanded_channels = make_divisible(
            int(round(in_channels * config.expand_ratio)),
            config.depth_divisible_by,
            config.min_depth,
        )

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            state_dict=state_dict,
            base_address=f"{base_address}.expand_1x1",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
            state_dict=state_dict,
            base_address=f"{base_address}.conv_3x3",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

        self.reduce_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
            state_dict=state_dict,
            base_address=f"{base_address}.reduce_1x1",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features

        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        return residual + features if self.use_residual else features


class MobileNetV2Stem(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        state_dict=None,
        base_address="",
        device=None,
        host=None,
        disable_conv_on_tt_device=True,
    ) -> None:
        super().__init__()

        # The very first layer is a regular 3x3 convolution with stride 2 that expands to 32 channels.
        # All other expansion layers use the expansion factor to compute the number of output channels.
        self.first_conv = MobileNetV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=2,
            state_dict=state_dict,
            base_address=f"{base_address}.first_conv",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

        if config.first_layer_is_expansion:
            self.expand_1x1 = None
        else:
            self.expand_1x1 = MobileNetV2ConvLayer(
                config,
                in_channels=expanded_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                state_dict=state_dict,
                base_address=f"{base_address}.expand_1x1",
                device=device,
                host=host,
                disable_conv_on_tt_device=disable_conv_on_tt_device,
            )

        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=1,
            groups=expanded_channels,
            state_dict=state_dict,
            base_address=f"{base_address}.conv_3x3",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

        self.reduce_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
            state_dict=state_dict,
            base_address=f"{base_address}.reduce_1x1",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.first_conv(features)
        if self.expand_1x1 is not None:
            features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return features


class MobileNetV2Model(nn.Module):
    def __init__(
        self,
        config,
        add_pooling_layer: bool = True,
        state_dict=None,
        base_address="",
        device=None,
        host=None,
        disable_conv_on_tt_device=True,
    ):
        super().__init__()
        self.base_address_with_dot = "" if base_address == "" else f"{base_address}."
        self.config = config

        # Output channels for the projection layers
        channels = [
            16,
            24,
            24,
            32,
            32,
            32,
            64,
            64,
            64,
            64,
            96,
            96,
            96,
            160,
            160,
            160,
            320,
        ]
        channels = [apply_depth_multiplier(config, x) for x in channels]

        # Strides for the depthwise layers
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        self.conv_stem = MobileNetV2Stem(
            config,
            in_channels=config.num_channels,
            expanded_channels=apply_depth_multiplier(config, 32),
            out_channels=channels[0],
            state_dict=state_dict,
            base_address=f"{self.base_address_with_dot}conv_stem",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

        current_stride = 2  # first conv layer has stride 2
        dilation = 1

        self.layer = nn.ModuleList()
        for i in range(16):
            # Keep making the feature maps smaller or use dilated convolution?
            if current_stride == config.output_stride:
                layer_stride = 1
                layer_dilation = dilation
                dilation *= strides[i]  # larger dilation starts in next block
            else:
                layer_stride = strides[i]
                layer_dilation = 1
                current_stride *= layer_stride

            self.layer.append(
                MobileNetV2InvertedResidual(
                    config,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=layer_stride,
                    dilation=layer_dilation,
                    state_dict=state_dict,
                    base_address=f"{self.base_address_with_dot}layer.{i}",
                    device=device,
                    host=host,
                    disable_conv_on_tt_device=disable_conv_on_tt_device,
                )
            )

        if config.finegrained_output and config.depth_multiplier < 1.0:
            output_channels = 1280
        else:
            output_channels = apply_depth_multiplier(config, 1280)

        self.conv_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=channels[-1],
            out_channels=output_channels,
            kernel_size=1,
            state_dict=state_dict,
            base_address=f"{self.base_address_with_dot}conv_1x1",
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> tuple:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.conv_stem(pixel_values)

        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        last_hidden_state = self.conv_1x1(hidden_states)

        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None

        return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)
