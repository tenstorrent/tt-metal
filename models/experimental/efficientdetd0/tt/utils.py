# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.tt_cnn.tt.builder import (
    TtConv2d,
    TtMaxPool2d,
    Conv2dConfiguration,
    UpsampleConfiguration,
    MaxPool2dConfiguration,
    HeightShardedStrategyConfiguration,
    WidthShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    AutoShardedStrategyConfiguration,
)
from models.experimental.efficientdetd0.tt.custom_preprocessor import UpsampleArgs

_sharding_map = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: WidthShardedStrategyConfiguration(reshard_if_not_optimal=True),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: BlockShardedStrategyConfiguration(reshard_if_not_optimal=True),
    None: AutoShardedStrategyConfiguration(),
}


class TtMaxPool2dDynamicSamePadding:
    def __init__(
        self,
        device,
        module_args,
        dtype=ttnn.bfloat16,
        deallocate_activation=False,
    ):
        self.device = device
        self.kernel_size = module_args.kernel_size
        self.stride = module_args.stride
        self.dilation = module_args.dilation
        self.input_height = module_args.input_height
        self.input_width = module_args.input_width
        ih, iw = self.input_height, self.input_width
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation + 1 - iw, 0)

        if self.pad_h > 0 or self.pad_w > 0:
            pad_offset_width = self.pad_w // 2 + self.pad_w - self.pad_w // 2
            pad_offset_height = self.pad_h // 2 + self.pad_h - self.pad_h // 2
            if pad_offset_width % 2 == 0 and pad_offset_height % 2 == 0:
                module_args.padding = (pad_offset_height // 2, pad_offset_width // 2)
            else:
                pad_top = pad_offset_height // 2
                pad_bottom = pad_top + pad_offset_height % 2
                pad_left = pad_offset_width // 2
                pad_right = pad_left + pad_offset_width % 2
                module_args.padding = (pad_top, pad_bottom, pad_left, pad_right)

        if isinstance(module_args.kernel_size, int):
            module_args.kernel_size = [module_args.kernel_size, module_args.kernel_size]
        if isinstance(module_args.stride, int):
            module_args.stride = [module_args.stride, module_args.stride]

        self.module_args = module_args
        maxpool_configuration = MaxPool2dConfiguration(
            input_height=module_args.input_height,
            input_width=module_args.input_width,
            channels=module_args.channels,
            batch_size=module_args.batch_size,
            kernel_size=module_args.kernel_size,
            stride=module_args.stride,
            padding=module_args.padding,
            ceil_mode=False,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_input=deallocate_activation,
            dtype=dtype,
        )
        self.dynamic_maxpool = TtMaxPool2d(
            configuration=maxpool_configuration,
            device=device,
        )

    def __call__(self, x):
        return self.dynamic_maxpool(x)


class TtConv2dDynamicSamePadding:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        module_args,
        dtype=ttnn.bfloat16,
        deallocate_activation=False,
    ):
        self.device = device
        self.parameters = parameters
        self.in_channels = module_args.in_channels
        self.kernel_size = module_args.kernel_size
        self.stride = module_args.stride
        self.dilation = module_args.dilation
        self.groups = module_args.groups
        self.input_height = module_args.input_height
        self.input_width = module_args.input_width
        self.shard_layout = shard_layout
        ih, iw = self.input_height, self.input_width
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if self.pad_h > 0 or self.pad_w > 0:
            pad_offset_width = self.pad_w // 2 + self.pad_w - self.pad_w // 2
            pad_offset_height = self.pad_h // 2 + self.pad_h - self.pad_h // 2
            if pad_offset_width % 2 == 0 and pad_offset_height % 2 == 0:
                module_args.padding = (pad_offset_height // 2, pad_offset_width // 2)
            else:
                pad_top = pad_offset_height // 2
                pad_bottom = pad_top + pad_offset_height % 2
                pad_left = pad_offset_width // 2
                pad_right = pad_left + pad_offset_width % 2
                module_args.padding = (pad_top, pad_bottom, pad_left, pad_right)

        self.module_args = module_args
        self.weights, self.bias = self.parameters["weight"], self.parameters["bias"]
        self.module_args.dtype = dtype
        conv_configuration = Conv2dConfiguration.from_model_args(
            conv2d_args=self.module_args,
            weights=self.weights,
            bias=self.bias,
            sharding_strategy=_sharding_map[shard_layout],
            deallocate_activation=deallocate_activation,
            enable_weights_double_buffer=False,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.dynamic_conv = TtConv2d(
            configuration=conv_configuration,
            device=device,
        )

    def __call__(self, x):
        return self.dynamic_conv(x)


class TtSeparableConvBlock:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        module_args,
        activation=False,
        deallocate_activation=False,
        dtype=ttnn.bfloat16,
    ):
        self.activation = activation
        self.depthwise_conv = TtConv2dDynamicSamePadding(
            device,
            parameters.depthwise_conv,
            shard_layout,
            module_args.depthwise_conv,
            deallocate_activation=deallocate_activation,
        )
        self.pointwise_conv = TtConv2dDynamicSamePadding(
            device,
            parameters.pointwise_conv,
            shard_layout,
            module_args.pointwise_conv,
            deallocate_activation=True,
            dtype=dtype,
        )

    def __call__(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.activation:
            x = x * ttnn.sigmoid_accurate(x, True)

        return x


class UpsampleConfiguration(UpsampleConfiguration):
    @classmethod
    def from_model_args(
        cls,
        upsample_args: UpsampleArgs,
        **kwargs,
    ):
        return cls(
            input_height=upsample_args.input_height,
            input_width=upsample_args.input_width,
            channels=upsample_args.channels,
            batch_size=upsample_args.batch_size,
            scale_factor=upsample_args.scale_factor,
            mode=upsample_args.mode,
            **kwargs,
        )
