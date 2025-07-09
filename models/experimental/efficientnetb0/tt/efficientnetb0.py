# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math


class EfficientNetb0Conv2D:
    def __init__(
        self,
        parameters,
        conv,
        device,
        cache={},
        activation="",
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        groups=1,
        output_layout=ttnn.TILE_LAYOUT,
        dilation=1,
    ):
        self.device = device
        self.batch_size = 1
        self.conv_params = conv
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.deallocate_activation = False
        self.cache = cache
        self.parameters = parameters
        self.shard_layout = shard_layout
        self.output_layout = output_layout
        self.shard_layout = shard_layout
        self.dilation = dilation
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=self.shard_layout,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=False,
        )

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def __call__(self, x):
        [x, [out_h, out_w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        return x


class Conv2dDynamicSamePadding:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        conv_params,
        batch=1,
        is_width_sharded=False,
    ):
        self.device = device
        self.batch = batch
        self.parameters = parameters
        self.in_channels = conv_params.in_channels
        self.kernel_size = conv_params.kernel_size
        self.stride = conv_params.stride
        self.dilation = conv_params.dilation
        self.groups = conv_params.groups
        self.input_height = conv_params.input_height
        self.input_width = conv_params.input_width
        self.shard_layout = shard_layout
        ih, iw = self.input_height, self.input_width
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if self.pad_h > 0 or self.pad_w > 0:
            conv_params.input_width = conv_params.input_width + self.pad_w // 2 + self.pad_w - self.pad_w // 2
            conv_params.input_height = conv_params.input_height + self.pad_h // 2 + self.pad_h - self.pad_h // 2
        if is_width_sharded:
            self.dynamic_conv = EfficientNetb0Conv2D(
                parameters,
                conv_params,
                device,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            )
        else:
            self.dynamic_conv = EfficientNetb0Conv2D(
                parameters,
                conv_params,
                device=device,
                shard_layout=self.shard_layout,
            )

        self.parameters_conv = conv_params

    def __call__(self, x):
        if self.pad_h > 0 or self.pad_w > 0:
            padded_shape = [self.batch, self.parameters_conv.input_height, self.parameters_conv.input_width, x.shape[3]]
            input_height = int(math.sqrt((x.shape[2] // self.batch)))
            input_width = int(math.sqrt((x.shape[2] // self.batch)))

            x = ttnn.sharded_to_interleaved(x)
            x = ttnn.reshape(x, (self.batch, input_height, input_width, x.shape[3]))
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.pad(
                x,
                padding=[
                    (0, 0),
                    ((padded_shape[1] - x.shape[1]) // 2, (padded_shape[1] - x.shape[1] + 1) // 2),
                    ((padded_shape[2] - x.shape[2]) // 2, (padded_shape[2] - x.shape[2] + 1) // 2),
                    (0, 0),
                ],
                value=0.0,
            )

        return self.dynamic_conv(x)


class MBConvBlock:
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
    ):
        self.parameters = parameters
        self.batch = batch
        self.conv_params = conv_params
        self.is_depthwise_first = is_depthwise_first
        self.is_height_sharded = is_height_sharded
        self.shard_layout = shard_layout
        if not is_depthwise_first:
            self._expand_conv = Conv2dDynamicSamePadding(
                device=device,
                parameters=parameters["_expand_conv"],
                conv_params=conv_params._expand_conv,
                shard_layout=self.shard_layout,
            )

        self._depthwise_conv = Conv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_depthwise_conv"],
            conv_params=conv_params._depthwise_conv,
            shard_layout=self.shard_layout,
        )

        self._se_reduce = Conv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_reduce"],
            conv_params=conv_params._se_reduce,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self._se_expand = Conv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_expand"],
            conv_params=conv_params._se_expand,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self._project_conv = Conv2dDynamicSamePadding(
            device,
            parameters=parameters["_project_conv"],
            conv_params=conv_params._project_conv,
            shard_layout=self.shard_layout,
        )

    def __call__(self, x):
        if not self.is_depthwise_first:
            x = self._expand_conv(x)
            x = x * ttnn.sigmoid_accurate(x)
        x = self._depthwise_conv(x)
        x = x * ttnn.sigmoid_accurate(x)
        mul1 = x

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.global_avg_pool2d(x)

        x = self._se_reduce(x)

        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        x = x * ttnn.sigmoid_accurate(x)
        x = self._se_expand(x)

        x = ttnn.sigmoid_accurate(x)
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
        self._conv_stem = Conv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_conv_stem"],
            conv_params=conv_params._conv_stem,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._blocks0 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks0"],
            is_depthwise_first=True,
            conv_params=conv_params._blocks0,
        )
        self._blocks1 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks1"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks1,
        )
        self._blocks2 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks2"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks2,
        )
        self._blocks3 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks3"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks3,
        )
        self._blocks4 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks4"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks4,
        )
        self._blocks5 = MBConvBlock(
            device, parameters["blocks"]["_blocks5"], is_depthwise_first=False, conv_params=conv_params._blocks5, id=5
        )
        self._blocks6 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks6"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks6,
            id=6,
        )
        self._blocks7 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks7"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks7,
            id=7,
        )
        self._blocks8 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks8"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks8,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=8,
        )
        self._blocks9 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks9"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks9,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=9,
        )
        self._blocks10 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks10"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks10,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=10,
        )
        self._blocks11 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks11"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks11,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id=11,
        )
        self._blocks12 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks12"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks12,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks13 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks13"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks13,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks14 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks14"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks14,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._blocks15 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks15"],
            is_depthwise_first=False,
            conv_params=conv_params._blocks15,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )
        self._conv_head = Conv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_conv_head"],
            conv_params=conv_params._conv_head,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )

        self.l1_weight = parameters["l1"]["weight"]
        self.l1_bias = parameters["l1"]["bias"]

    def __call__(self, x):
        x = self._conv_stem(x)
        x = x * ttnn.sigmoid_accurate(x)

        x = self._blocks0(x)

        x_1 = self._blocks1(x)
        x = self._blocks2(x_1)

        x = ttnn.add(x, x_1)
        x_3 = self._blocks3(x)
        x = self._blocks4(x_3)

        x = x + x_3
        x_5 = self._blocks5(x)
        x = self._blocks6(x_5)

        x_7_in = x + x_5
        x = self._blocks7(x_7_in)

        x = x_7_in + x
        x_8 = self._blocks8(x)
        x = self._blocks9(x_8)

        x_10_in = x + x_8
        x = self._blocks10(x_10_in)

        x = x + x_10_in
        x_11 = self._blocks11(x)
        x = self._blocks12(x_11)

        x_13_in = x + x_11
        x = self._blocks13(x_13_in)

        x_14_in = x + x_13_in
        x = self._blocks14(x_14_in)

        x = x_14_in + x
        x = self._blocks15(x)
        x = self._conv_head(x)

        x = x * ttnn.sigmoid(x)

        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.global_avg_pool2d(x)

        x = ttnn.reshape(x, (1, -1))

        x = ttnn.linear(x, self.l1_weight, bias=self.l1_bias)

        return x
