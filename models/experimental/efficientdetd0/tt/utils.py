# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.experimental.efficientnetb0.tt.efficientnetb0 import Conv2dDynamicSamePadding


class EfficientDetd0Maxpool2D:
    def __init__(
        self,
        maxpool_params,
        device,
        cache={},
        dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=ttnn.TILE_LAYOUT,
        dilation=1,
        ceil_mode=False,
        memory_config=None,
        in_place_halo=False,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch_size = maxpool_params.batch_size
        self.input_height = maxpool_params.input_height
        self.input_width = maxpool_params.input_width
        self.channels = maxpool_params.channels
        self.kernel_size = maxpool_params.kernel_size
        self.padding = maxpool_params.padding
        self.stride = maxpool_params.stride
        self.ceil_mode = maxpool_params.ceil_mode
        # self.ceil_mode = ceil_mode
        self.deallocate_activation = deallocate_activation
        self.cache = cache
        self.dtype = dtype
        self.memory_config = (memory_config,)
        self.in_place_halo = (in_place_halo,)
        self.shard_layout = shard_layout
        self.output_layout = output_layout
        self.dilation = dilation

    def __call__(self, x):
        return ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=self.input_height,
            input_w=self.input_width,
            channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=(self.dilation, self.dilation),
            ceil_mode=self.ceil_mode,
            memory_config=self.memory_config,
            applied_shard_scheme=self.shard_layout,
            in_place_halo=self.in_place_halo,
            deallocate_input=self.deallocate_activation,
            reallocate_halo_output=True,
            dtype=self.dtype,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )


class MaxPool2dDynamicSamePadding:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        maxpool_params,
        batch=1,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch = batch
        self.parameters = parameters
        self.in_channels = maxpool_params.in_channels
        self.kernel_size = maxpool_params.kernel_size
        self.stride = maxpool_params.stride
        self.dilation = maxpool_params.dilation
        self.groups = maxpool_params.groups
        self.input_height = maxpool_params.input_height
        self.input_width = maxpool_params.input_width
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
                maxpool_params.padding = (pad_offset_height // 2, pad_offset_width // 2)
            else:
                pad_top = pad_offset_height // 2
                pad_bottom = pad_top + pad_offset_height % 2
                pad_left = pad_offset_width // 2
                pad_right = pad_left + pad_offset_width % 2
                maxpool_params.padding = (pad_top, pad_bottom, pad_left, pad_right)

        self.dynamic_conv = EfficientDetd0Maxpool2D(
            maxpool_params,
            device=device,
            shard_layout=self.shard_layout,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, x):
        return self.dynamic_conv(x)


class SeparableConvBlock:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        conv_params,
        activation=False,
        batch=1,
        deallocate_activation=False,
    ):
        self.activation = activation
        self.depthwise_conv = Conv2dDynamicSamePadding(
            device,
            parameters.depthwise_conv,
            shard_layout,
            conv_params.depthwise_conv,
            batch=batch,
            deallocate_activation=deallocate_activation,
        )
        self.pointwise_conv = Conv2dDynamicSamePadding(
            device,
            parameters.pointwise_conv,
            shard_layout,
            conv_params.pointwise_conv,
            batch=batch,
            deallocate_activation=True,
        )

    def __call__(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.activation:
            x = x * ttnn.sigmoid_accurate(x, True)

        return x
