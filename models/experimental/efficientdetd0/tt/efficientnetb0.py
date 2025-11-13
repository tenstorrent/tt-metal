# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.efficientdetd0.tt.utils import Conv2dDynamicSamePadding as TtConv2dDynamicSamePadding


class TtnnMBConvBlock:
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
        id_skip=True,
    ):
        self.parameters = parameters
        self.batch = batch
        self.conv_params = conv_params
        self.is_depthwise_first = is_depthwise_first
        self.is_height_sharded = is_height_sharded
        self.shard_layout = shard_layout
        self.in_features = conv_params._depthwise_conv.in_channels
        if not is_depthwise_first:
            self.in_features = conv_params._expand_conv.in_channels
        self.out_features = conv_params._project_conv.out_channels
        self.id_skip = (
            id_skip and (conv_params._depthwise_conv.stride[0] == 1) and self.in_features == self.out_features
        )

        if not is_depthwise_first:
            self._expand_conv = TtConv2dDynamicSamePadding(
                device=device,
                parameters=parameters["_expand_conv"],
                conv_params=conv_params._expand_conv,
                shard_layout=self.shard_layout,
                deallocate_activation=deallocate_activation,
            )

        self._depthwise_conv = TtConv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_depthwise_conv"],
            conv_params=conv_params._depthwise_conv,
            shard_layout=shard_layout_depthwise_conv,
            deallocate_activation=deallocate_activation,
        )

        self._se_reduce = TtConv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_reduce"],
            conv_params=conv_params._se_reduce,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=deallocate_activation,
        )

        self._se_expand = TtConv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_expand"],
            conv_params=conv_params._se_expand,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=deallocate_activation,
        )

        self._project_conv = TtConv2dDynamicSamePadding(
            device,
            parameters=parameters["_project_conv"],
            conv_params=conv_params._project_conv,
            shard_layout=self.shard_layout,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, x):
        if self.id_skip:
            if x.is_sharded():
                skip_input = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            else:
                skip_input = ttnn.clone(x)
                skip_input = ttnn.to_memory_config(skip_input, ttnn.DRAM_MEMORY_CONFIG)

        if not self.is_depthwise_first:
            x = self._expand_conv(x)
            x = x * ttnn.sigmoid_accurate(x, True)

        x = self._depthwise_conv(x)
        x = x * ttnn.sigmoid_accurate(x, True)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        if x.shape[-1] != 32 and x.shape[-1] != 96:
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

        x_squeezed = ttnn.global_avg_pool2d(x)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = x_squeezed * ttnn.sigmoid_accurate(x_squeezed, True)
        x_squeezed = self._se_expand(x_squeezed)

        if x_squeezed.is_sharded():
            x_squeezed = ttnn.sharded_to_interleaved(x_squeezed, ttnn.L1_MEMORY_CONFIG)

        x = ttnn.sigmoid_accurate(x_squeezed, True) * x
        ttnn.deallocate(x_squeezed)

        x = self._project_conv(x)

        if self.id_skip:
            x = x + skip_input
            ttnn.deallocate(skip_input)

        return x


class TtnnEfficientNet:
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
        self._blocks0 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks0"],
            is_depthwise_first=True,
            conv_params=conv_params._blocks[0],
            deallocate_activation=True,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._blocks1 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks1"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[1],
            deallocate_activation=True,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._blocks2 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks2"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[2],
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._blocks3 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks3"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[3],
            deallocate_activation=True,
        )
        self._blocks4 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks4"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[4],
        )
        self._blocks5 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks5"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[5],
            id=5,
            deallocate_activation=True,
        )
        self._blocks6 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks6"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[6],
            id=6,
        )
        self._blocks7 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks7"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[7],
            id=7,
        )
        self._blocks8 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks8"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[8],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=8,
            deallocate_activation=True,
        )
        self._blocks9 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks9"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[9],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=9,
        )
        self._blocks10 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks10"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[10],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=10,
        )
        self._blocks11 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks11"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[11],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=11,
            deallocate_activation=True,
        )
        self._blocks12 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks12"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[12],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks13 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks13"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[13],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks14 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks14"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[14],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks15 = TtnnMBConvBlock(
            device,
            parameters["blocks"]["_blocks15"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks[15],
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_layout_depthwise_conv=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
        )

    def __call__(self, x):
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = self._conv_stem(x)
        x = ttnn.swish(x)
        x = self._blocks0(x)
        x = self._blocks1(x)
        x = self._blocks2(x)
        x = self._blocks3(x)
        x = self._blocks4(x)

        if x.is_sharded():
            x_4 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        else:
            x_4 = ttnn.clone(x)
            x_4 = ttnn.to_memory_config(x_4, ttnn.DRAM_MEMORY_CONFIG)

        x = self._blocks5(x)
        x = self._blocks6(x)
        x = self._blocks7(x)
        x = self._blocks8(x)
        x = self._blocks9(x)
        x = self._blocks10(x)

        if x.is_sharded():
            x_10 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        else:
            x_10 = ttnn.clone(x)
            x_10 = ttnn.to_memory_config(x_10, ttnn.DRAM_MEMORY_CONFIG)

        x = self._blocks11(x)
        x = self._blocks12(x)
        x = self._blocks13(x)
        x = self._blocks14(x)
        x = self._blocks15(x)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        return x_4, x_10, x
