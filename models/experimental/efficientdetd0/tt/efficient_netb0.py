# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.efficientdetd0.tt.utils import Conv2dDynamicSamePadding as TtConv2dDynamicSamePadding


class Efficientdetd0MBConvBlock:
    def __init__(
        self,
        device,
        parameters,
        conv_params,
        batch=1,
        is_depthwise_first=False,
        is_height_sharded=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        id=1,
        deallocate_activation=False,
        shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ):
        self.parameters = parameters
        self.batch = batch
        self.conv_params = conv_params
        self.is_depthwise_first = is_depthwise_first
        self.is_height_sharded = is_height_sharded
        self.shard_layout = shard_layout
        if not is_depthwise_first:
            self._expand_conv = TtConv2dDynamicSamePadding(
                device=device,
                parameters=parameters["_expand_conv"],
                conv_params=conv_params._expand_conv,
                shard_layout=self.shard_layout,
                deallocate_activation=deallocate_activation,
                dtype=ttnn.bfloat8_b,
            )

        self._depthwise_conv = TtConv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_depthwise_conv"],
            conv_params=conv_params._depthwise_conv,
            shard_layout=shard_layout_depthwise_conv,
            deallocate_activation=deallocate_activation,
            dtype=ttnn.bfloat8_b,
        )

        self._se_reduce = TtConv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_reduce"],
            conv_params=conv_params._se_reduce,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=deallocate_activation,
            dtype=ttnn.bfloat8_b,
        )

        self._se_expand = TtConv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_expand"],
            conv_params=conv_params._se_expand,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=deallocate_activation,
            dtype=ttnn.bfloat8_b,
        )

        self._project_conv = TtConv2dDynamicSamePadding(
            device,
            parameters=parameters["_project_conv"],
            conv_params=conv_params._project_conv,
            shard_layout=self.shard_layout,
            deallocate_activation=deallocate_activation,
            dtype=ttnn.bfloat8_b,
        )

    def __call__(self, x):
        if not self.is_depthwise_first:
            x = self._expand_conv(x)
            x = x * ttnn.sigmoid_accurate(x, True)
        x = self._depthwise_conv(x)
        x = x * ttnn.sigmoid_accurate(x, True)
        mul1 = x

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        if x.shape[-1] != 32 and x.shape[-1] != 96:
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.global_avg_pool2d(x)

        x = self._se_reduce(x)

        x = x * ttnn.sigmoid_accurate(x, True)
        x = self._se_expand(x)

        x = ttnn.sigmoid_accurate(x, True)
        mul1_interleaved = mul1
        if mul1.is_sharded():
            mul1_interleaved = ttnn.sharded_to_interleaved(mul1, ttnn.L1_MEMORY_CONFIG)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mul1)
        x = x * mul1_interleaved

        x = self._project_conv(x)

        return x


class Efficientnetb0:
    def __init__(self, device, parameters, conv_params, batch=1):
        self.device = device
        self.parameters = parameters
        self.batch = batch
        self.conv_cache = {}
        self.conv_params = conv_params
        self._conv_stem = TtConv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_conv_stem"],
            conv_params=conv_params._conv_stem,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
        )
        self._blocks0 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks0"],
            is_depthwise_first=True,
            conv_params=conv_params._blocks0,
            deallocate_activation=True,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._blocks1 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks1"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks1,
            deallocate_activation=True,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._blocks2 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks2"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks2,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._blocks3 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks3"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks3,
            deallocate_activation=True,
        )
        self._blocks4 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks4"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks4,
        )
        self._blocks5 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks5"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks5,
            id=5,
            deallocate_activation=True,
        )
        self._blocks6 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks6"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks6,
            id=6,
        )
        self._blocks7 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks7"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks7,
            id=7,
        )
        self._blocks8 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks8"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks8,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=8,
            deallocate_activation=True,
        )
        self._blocks9 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks9"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks9,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=9,
        )
        self._blocks10 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks10"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks10,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=10,
        )
        self._blocks11 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks11"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks11,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=11,
            deallocate_activation=True,
        )
        self._blocks12 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks12"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks12,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks13 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks13"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks13,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks14 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks14"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks14,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks15 = Efficientdetd0MBConvBlock(
            device,
            parameters["blocks"]["_blocks15"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks15,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
        )

    def __call__(self, x):
        N, C, H, W = x.shape
        min_channels = 16  # Padding from image channels (3) to min channels (16)
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(x, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = x
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
        ttnn.deallocate(nchw)
        ttnn.deallocate(x)
        nhwc = ttnn.reallocate(nhwc)
        x = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])

        x = self._conv_stem(x)

        x = ttnn.swish(x)

        x = self._blocks0(x)

        x_1 = self._blocks1(x)

        x = self._blocks2(x_1)

        x = ttnn.add(x, x_1)
        ttnn.deallocate(x_1)

        x_3 = self._blocks3(x)

        x = self._blocks4(x_3)

        x = x + x_3
        ttnn.deallocate(x_3)

        x_5 = self._blocks5(x)

        x = self._blocks6(x_5)

        x_7_in = x + x_5
        ttnn.deallocate(x_5)

        x = self._blocks7(x_7_in)

        x = x_7_in + x
        ttnn.deallocate(x_7_in)

        x_8 = self._blocks8(x)

        x = self._blocks9(x_8)

        x_10_in = x + x_8
        ttnn.deallocate(x_8)

        x = self._blocks10(x_10_in)

        x = x + x_10_in
        ttnn.deallocate(x_10_in)

        x_11 = self._blocks11(x)

        x = self._blocks12(x_11)

        x_13_in = x + x_11
        ttnn.deallocate(x_11)

        x = self._blocks13(x_13_in)

        x_14_in = x + x_13_in
        ttnn.deallocate(x_13_in)
        ttnn.deallocate(x)

        x = self._blocks14(x_14_in)

        x = x_14_in + x
        ttnn.deallocate(x_14_in)

        x = self._blocks15(x)

        return x_3, x_8, x
