# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from ttnn.model_preprocessing import ParameterDict, fold_batch_norm2d_into_conv2d
from torch import nn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
import math
import torch.nn.functional as F


def input_preprocessing(x, N, C, H, W):
    x = ttnn.to_torch(x)
    x = torch.permute(x, (0, 3, 1, 2))
    x = x.reshape(N, C, H, W)
    return x


class EfficientNetb0Conv2D:
    def __init__(
        self,
        conv,
        conv_pth=None,
        bn_pth=None,
        bn=None,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.device = device
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = conv.use_1d_systolic_array
        self.deallocate_activation = False
        self.cache = cache

        self.shard_layout = shard_layout
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            math_fidelity=ttnn.MathFidelity.LoFi,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate_activation,
            activation=activation,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if bn is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(conv_pth, bn_pth)
        else:
            weight, bias = conv_pth.weight, conv_pth.bias

        weight = weight
        bias = torch.reshape(bias, (1, 1, 1, -1))

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)
        self.bias = ttnn.from_torch(bias, dtype=ttnn.float32)

    def __call__(self, x):
        x, _, _, self.weight, self.bias = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            conv_op_cache=self.cache,
            groups=self.groups,
        )
        return x


class Conv2dDynamicSamePadding:
    def __init__(
        self, device, parameters_conv, parameter_bn=None, conv_pth=None, bn_pth=None, batch=1, is_width_sharded=False
    ):
        self.device = device
        self.batch = batch
        self.in_channels = parameters_conv.in_channels
        self.kernel_size = parameters_conv.kernel_size
        self.stride = parameters_conv.stride
        self.dilation = parameters_conv.dilation
        self.groups = parameters_conv.groups
        self.input_height = parameters_conv.input_height
        self.input_width = parameters_conv.input_width
        self.parameters_conv = parameters_conv

        ih, iw = self.input_height, self.input_width
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if self.pad_h > 0 or self.pad_w > 0:
            parameters_conv.input_width = parameters_conv.input_width + self.pad_w // 2 + self.pad_w - self.pad_w // 2
            parameters_conv.input_height = parameters_conv.input_height + self.pad_h // 2 + self.pad_h - self.pad_h // 2

        if is_width_sharded:
            self.dynamic_conv = EfficientNetb0Conv2D(
                parameters_conv,
                conv_pth=conv_pth,
                bn_pth=bn_pth,
                bn=parameter_bn,
                device=device,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            )
        else:
            self.dynamic_conv = EfficientNetb0Conv2D(
                parameters_conv, conv_pth=conv_pth, bn_pth=bn_pth, bn=parameter_bn, device=device
            )

    def __call__(self, x):
        if self.pad_h > 0 or self.pad_w > 0:
            x = input_preprocessing(x, self.batch, self.in_channels, self.input_height, self.input_width)

            x = F.pad(x, [self.pad_w // 2, self.pad_w - self.pad_w // 2, self.pad_h // 2, self.pad_h - self.pad_h // 2])

            x = torch.permute(x, (0, 2, 3, 1))
            x = x.reshape(
                1,
                1,
                self.batch * self.parameters_conv.input_height * self.parameters_conv.input_width,
                self.parameters_conv.in_channels,
            )

            x = ttnn.from_torch(x, dtype=ttnn.bfloat16)

        return self.dynamic_conv(x)


class MBConvBlock:
    def __init__(self, device, parameters, batch=1, is_depthwise_first=False, is_width_sharded=False):
        self.parameters = parameters
        self.batch = batch
        self.is_depthwise_first = is_depthwise_first
        self.is_width_sharded = is_width_sharded
        if not is_depthwise_first:
            self._expand_conv = Conv2dDynamicSamePadding(
                device,
                parameters._expand_conv,
                parameter_bn=parameters._bn0,
                conv_pth=parameters.module._expand_conv,
                bn_pth=parameters.module._bn0,
            )

        self._depthwise_conv = Conv2dDynamicSamePadding(
            device,
            parameters._depthwise_conv,
            parameter_bn=parameters._bn1,
            conv_pth=parameters.module._depthwise_conv,
            bn_pth=parameters.module._bn1,
            is_width_sharded=is_width_sharded,
        )

        self._se_reduce = Conv2dDynamicSamePadding(device, parameters._se_reduce, conv_pth=parameters.module._se_reduce)

        self._se_expand = Conv2dDynamicSamePadding(device, parameters._se_expand, conv_pth=parameters.module._se_expand)

        self._project_conv = Conv2dDynamicSamePadding(
            device,
            parameters._project_conv,
            parameter_bn=parameters._bn2,
            conv_pth=parameters.module._project_conv,
            bn_pth=parameters.module._bn2,
        )

        self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)

    def __call__(self, x):
        if not self.is_depthwise_first:
            x = self._expand_conv(x)
            x = x * ttnn.sigmoid(x)

        x = self._depthwise_conv(x)
        tt_out = input_preprocessing(
            x,
            self.batch,
            self.parameters._depthwise_conv.out_channels,
            int(math.sqrt(x.shape[2])),
            int(math.sqrt(x.shape[2])),
        )
        print(x.shape)
        torch.save(tt_out, "tt_out.pt")
        if x.is_sharded() and self.is_width_sharded:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        print(x.memory_config())
        x = x * ttnn.sigmoid(x)
        mul1 = x
        x = input_preprocessing(
            x,
            self.batch,
            self.parameters._depthwise_conv.out_channels,
            int(math.sqrt(x.shape[2])),
            int(math.sqrt(x.shape[2])),
        )

        x = self._avg_pooling(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(
            1,
            1,
            1,
            self.parameters._se_reduce.in_channels,
        )
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16)

        x = self._se_reduce(x)
        x = x * ttnn.sigmoid(x)
        x = self._se_expand(x)
        x = ttnn.sigmoid(x)
        mul1_interleaved = mul1
        if mul1.is_sharded():
            mul1_interleaved = ttnn.sharded_to_interleaved(mul1, ttnn.L1_MEMORY_CONFIG)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        x = x * mul1_interleaved

        x = self._project_conv(x)

        return x


class Efficientnetb0:
    def __init__(self, device, parameters, torch_model, batch=1):
        self.device = device
        self.parameters = parameters
        self.batch = batch
        self.conv_cache = {}
        self._conv_stem = Conv2dDynamicSamePadding(
            device,
            parameters._conv_stem,
            parameter_bn=parameters._bn0,
            conv_pth=parameters._conv_stem.module,
            bn_pth=parameters._bn0.module,
        )
        self._blocks0 = MBConvBlock(
            device,
            parameters._blocks0,
            is_depthwise_first=True,
        )
        self._blocks1 = MBConvBlock(
            device,
            parameters._blocks1,
            is_depthwise_first=False,
        )
        self._blocks2 = MBConvBlock(
            device,
            parameters._blocks2,
            is_depthwise_first=False,
        )
        self._blocks3 = MBConvBlock(
            device,
            parameters._blocks3,
            is_depthwise_first=False,
        )
        self._blocks4 = MBConvBlock(
            device,
            parameters._blocks4,
            is_depthwise_first=False,
        )
        self._blocks5 = MBConvBlock(
            device,
            parameters._blocks5,
            is_depthwise_first=False,
        )
        self._blocks6 = MBConvBlock(
            device,
            parameters._blocks6,
            is_depthwise_first=False,
        )
        self._blocks7 = MBConvBlock(
            device,
            parameters._blocks7,
            is_depthwise_first=False,
        )
        self._blocks8 = MBConvBlock(
            device,
            parameters._blocks8,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._blocks9 = MBConvBlock(
            device,
            parameters._blocks9,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._blocks10 = MBConvBlock(
            device,
            parameters._blocks10,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._blocks11 = MBConvBlock(
            device,
            parameters._blocks11,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._blocks12 = MBConvBlock(
            device,
            parameters._blocks12,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._blocks13 = MBConvBlock(
            device,
            parameters._blocks13,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._blocks14 = MBConvBlock(
            device,
            parameters._blocks14,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._blocks15 = MBConvBlock(
            device,
            parameters._blocks15,
            is_depthwise_first=False,
            is_width_sharded=True,
        )
        self._conv_head = Conv2dDynamicSamePadding(
            device,
            parameters._conv_head,
            parameter_bn=parameters._bn1,
            conv_pth=parameters._conv_head.module,
            bn_pth=parameters._bn1.module,
        )
        self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.l1_weight = parameters._fc.module.weight
        self.l1_bias = parameters._fc.module.bias

    def __call__(self, x):
        x = self._conv_stem(x)
        x = ttnn.to_device(x, device=self.device)
        x = x * ttnn.sigmoid(x)
        x = self._blocks0(x)
        x_1 = self._blocks1(x)
        x = self._blocks2(x_1)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        x = ttnn.add(x, x_1)
        x_3 = self._blocks3(x)
        x = self._blocks4(x_3)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        if x_3.is_sharded():
            x_3 = ttnn.sharded_to_interleaved(x_3, ttnn.L1_MEMORY_CONFIG)

        x = x + x_3
        x_5 = self._blocks5(x)
        x = self._blocks6(x_5)

        if x_5.is_sharded():
            x_5 = ttnn.sharded_to_interleaved(x_5, ttnn.L1_MEMORY_CONFIG)

        x_7_in = x + x_5
        x = self._blocks7(x_7_in)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        if x_7_in.is_sharded():
            x_7_in = ttnn.sharded_to_interleaved(x_7_in, ttnn.L1_MEMORY_CONFIG)

        x = x_7_in + x
        x_8 = self._blocks8(x)
        print("block9")
        x = self._blocks9(x_8)

        # if(x.is_sharded()):
        #     x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # if(x_8.is_sharded()):
        #     x_8 = ttnn.sharded_to_interleaved(x_8, ttnn.L1_MEMORY_CONFIG)

        print("block10")
        x_10_in = x + x_8
        x = self._blocks10(x_10_in)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        x = x + x_10_in
        x_11 = self._blocks11(x)
        x = self._blocks12(x_11)

        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x_11 = ttnn.sharded_to_interleaved(x_11, ttnn.L1_MEMORY_CONFIG)

        x_13_in = x + x_11
        x = self._blocks13(x_13_in)

        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        x_14_in = x + x_13_in
        x = self._blocks14(x)

        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        x = x_14_in + x
        x = self._blocks15(x)
        x = self._conv_head(x)
        x = x * ttnn.sigmoid(x)

        x = input_preprocessing(
            x,
            self.batch,
            self.parameters._conv_head.out_channels,
            int(math.sqrt(x.shape[2])),
            int(math.sqrt(x.shape[2])),
        )

        x = self._avg_pooling(x)

        x = torch.flatten(x, 1)

        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        self.l1_weight = preprocess_linear_weight(self.l1_weight, dtype=ttnn.bfloat16)
        self.l1_bias = preprocess_linear_bias(self.l1_bias, dtype=ttnn.bfloat16)
        self.l1_weight = ttnn.to_device(self.l1_weight, self.device)
        self.l1_bias = ttnn.to_device(self.l1_bias, self.device)

        x = ttnn.linear(x, self.l1_weight, bias=self.l1_bias)

        return ttnn.from_device(x)
