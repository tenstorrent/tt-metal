# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

from typing import Union
from dataclasses import replace
from models.tt_cnn.tt.builder import (
    TtConv2d,
    TtMaxPool2d,
    Conv2dConfiguration,
    UpsampleConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
)
from models.experimental.efficientdetd0.tt.custom_preprocessor import UpsampleArgs, MaxPool2dArgs, Conv2dArgs


def generate_conv_configuration_from_args(conv2d_args: Conv2dArgs, parameters_dict: dict, **kwargs):
    packer_l1_acc = kwargs.pop("packer_l1_acc", True)
    fp32_dest_acc_en = kwargs.pop("fp32_dest_acc_en", True)
    enable_weights_double_buffer = kwargs.pop("fp32_dest_acc_en", False)
    math_fidelity = kwargs.pop("math_fidelity", ttnn.MathFidelity.HiFi2)
    weights, bias = parameters_dict["weight"], parameters_dict["bias"]

    return Conv2dConfiguration.from_model_args(
        conv2d_args=conv2d_args,
        weights=weights,
        bias=bias,
        enable_weights_double_buffer=enable_weights_double_buffer,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        **kwargs,
    )


def generate_maxpool_configuration_from_args(maxpool2d_args: MaxPool2dArgs, **kwargs):
    return MaxPool2dConfiguration(
        input_height=maxpool2d_args.input_height,
        input_width=maxpool2d_args.input_width,
        channels=maxpool2d_args.channels,
        batch_size=maxpool2d_args.batch_size,
        kernel_size=maxpool2d_args.kernel_size,
        stride=maxpool2d_args.stride,
        padding=maxpool2d_args.padding,
        **kwargs,
    )


def generate_upsample_configuration_from_args(upsample_args: UpsampleArgs, **kwargs):
    return UpsampleConfiguration(
        input_height=upsample_args.input_height,
        input_width=upsample_args.input_width,
        channels=upsample_args.channels,
        batch_size=upsample_args.batch_size,
        scale_factor=upsample_args.scale_factor,
        mode=upsample_args.mode,
        **kwargs,
    )


def _get_dynamic_padding(configuration: Union[MaxPool2dConfiguration, Conv2dConfiguration]):
    """
    Helper function for calulating the dynamic padding for given input shapes; similar to Tensorflow implementation
    """
    assert len(configuration.kernel_size) == len(configuration.stride) == len(configuration.dilation) == 2

    ih, iw = configuration.input_height, configuration.input_width
    kh, kw = configuration.kernel_size
    sh, sw = configuration.stride
    dh, dw = configuration.dilation
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
    pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
    pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)

    padding = configuration.padding
    if pad_h > 0 or pad_w > 0:
        pad_offset_width = pad_w // 2 + pad_w - pad_w // 2
        pad_offset_height = pad_h // 2 + pad_h - pad_h // 2
        if pad_offset_width % 2 == 0 and pad_offset_height % 2 == 0:
            padding = [pad_offset_height // 2, pad_offset_width // 2]
        else:
            pad_top = pad_offset_height // 2
            pad_bottom = pad_top + pad_offset_height % 2
            pad_left = pad_offset_width // 2
            pad_right = pad_left + pad_offset_width % 2
            padding = [pad_top, pad_bottom, pad_left, pad_right]

    return padding


class TtMaxPool2dDynamicSamePadding(TtMaxPool2d):
    def __init__(
        self,
        configuration: MaxPool2dConfiguration,
        device: ttnn.Device,
    ):
        super().__init__(configuration, device)
        self.configuration = replace(self.configuration, padding=_get_dynamic_padding(self.configuration))


class TtConv2dDynamicSamePadding(TtConv2d):
    def __init__(
        self,
        configuration: Conv2dConfiguration,
        device: ttnn.Device,
    ):
        super().__init__(configuration, device)
        self.configuration = replace(self.configuration, padding=_get_dynamic_padding(self.configuration))


class TtSeparableConvBlock:
    def __init__(
        self,
        device,
        parameters,
        module_args,
        sharding_strategy=AutoShardedStrategyConfiguration(),
        activation=False,
        deallocate_activation=False,
        dtype=ttnn.bfloat16,
    ):
        self.activation = activation
        self.depthwise_conv = TtConv2dDynamicSamePadding(
            configuration=generate_conv_configuration_from_args(
                conv2d_args=module_args.depthwise_conv,
                parameters_dict=parameters.depthwise_conv,
                sharding_strategy=sharding_strategy,
                deallocate_activation=deallocate_activation,
            ),
            device=device,
        )
        module_args.pointwise_conv.dtype = dtype
        self.pointwise_conv = TtConv2dDynamicSamePadding(
            configuration=generate_conv_configuration_from_args(
                conv2d_args=module_args.pointwise_conv,
                parameters_dict=parameters.pointwise_conv,
                sharding_strategy=sharding_strategy,
                deallocate_activation=True,
            ),
            device=device,
        )

    def __call__(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.activation:
            x = x * ttnn.sigmoid_accurate(x, True)

        return x
