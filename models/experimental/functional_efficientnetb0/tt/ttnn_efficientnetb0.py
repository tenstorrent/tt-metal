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
        parameters,
        input_params,
        device,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        groups=1,
        use_shallow_covariant=False,
        output_layout=ttnn.TILE_LAYOUT,
        dilation=1,
    ):
        self.device = device
        self.batch_size = 1
        self.input_params = input_params
        self.groups = groups
        self.deallocate_activation = False
        self.cache = cache
        self.parameters = parameters
        self.shard_layout = shard_layout
        self.use_shallow_covariant = use_shallow_covariant
        self.activation_dtype = activation_dtype
        self.output_layout = output_layout
        self.shard_layout = shard_layout
        self.dilation = dilation
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=self.activation_dtype,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=self.shard_layout,
            input_channels_alignment=16 if self.use_shallow_covariant else 32,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=True,
        )

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt((x.shape[2] // self.batch_size)))
            input_width = int(math.sqrt((x.shape[2] // self.batch_size)))
        [x, [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=x.shape[3],
            out_channels=self.input_params[3],
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            conv_op_cache=self.cache,
            debug=False,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=False,
        )

        return x


class Conv2dDynamicSamePadding:
    def __init__(
        self, device, parameters, input_params, groups=1, batch=1, is_width_sharded=False, use_shallow_covariant=False
    ):
        self.device = device
        self.batch = batch
        self.groups = groups
        self.input_params = input_params
        self.use_shallow_covariant = use_shallow_covariant

        # ih, iw = self.input_height, self.input_width
        # kh, kw = self.kernel_size
        # sh, sw = self.stride
        # oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        # self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        # self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        # if self.pad_h > 0 or self.pad_w > 0:
        #     parameters_conv.input_width = parameters_conv.input_width + self.pad_w // 2 + self.pad_w - self.pad_w // 2
        #     parameters_conv.input_height = parameters_conv.input_height + self.pad_h // 2 + self.pad_h - self.pad_h // 2

        if is_width_sharded:
            self.dynamic_conv = EfficientNetb0Conv2D(
                parameters,
                input_params=self.input_params,
                device=device,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                use_shallow_covariant=self.use_shallow_covariant,
                groups=self.groups,
            )
        else:
            self.dynamic_conv = EfficientNetb0Conv2D(
                parameters,
                input_params=self.input_params,
                device=device,
                use_shallow_covariant=self.use_shallow_covariant,
                groups=self.groups,
            )

    def __call__(self, x):
        if x.shape[1] != 1:
            i_h = x.shape[1]
            i_w = x.shape[2]
        else:
            i_h = int(math.sqrt((x.shape[2] // self.batch)))
            i_w = int(math.sqrt((x.shape[2] // self.batch)))
        kh, kw = (self.input_params[0], self.input_params[0])
        sh, sw = (self.input_params[1], self.input_params[1])
        oh, ow = math.ceil(i_h / sh), math.ceil(i_w / sw)  # change the output size according to stride ! ! !
        self.pad_h = max((oh - 1) * sh + (kh - 1) * 1 + 1 - i_h, 0)
        self.pad_w = max((ow - 1) * sw + (kw - 1) * 1 + 1 - i_w, 0)
        if self.pad_h > 0 or self.pad_w > 0:
            input_width = i_w + self.pad_w // 2 + self.pad_w - self.pad_w // 2
            input_height = i_h + self.pad_h // 2 + self.pad_h - self.pad_h // 2
        if self.pad_h > 0 or self.pad_w > 0:
            x = input_preprocessing(x, self.batch, x.shape[3], i_h, i_w)

            x = F.pad(x, [self.pad_w // 2, self.pad_w - self.pad_w // 2, self.pad_h // 2, self.pad_h - self.pad_h // 2])

            x = torch.permute(x, (0, 2, 3, 1))
            x = x.reshape(
                1,
                1,
                self.batch * input_width * input_height,
                x.shape[3],
            )

            x = ttnn.from_torch(x, dtype=ttnn.bfloat16)

        return self.dynamic_conv(x)


class MBConvBlock:
    def __init__(
        self, device, parameters, input_params, groups=1, batch=1, is_depthwise_first=False, is_width_sharded=False
    ):
        self.parameters = parameters
        self.batch = batch
        self.input_params = input_params
        self.groups = groups
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
            device=device,
            parameters=parameters["_depthwise_conv"],
            input_params=self.input_params[0],
            groups=self.groups,
        )

        self._se_reduce = Conv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_reduce"],
            input_params=self.input_params[1],
        )

        self._se_expand = Conv2dDynamicSamePadding(
            device=device,
            parameters=parameters["_se_reduce"],
            input_params=self.input_params[2],
        )

        self._project_conv = Conv2dDynamicSamePadding(
            device,
            parameters=parameters["_project_conv"],
            input_params=self.input_params[3],
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
            device=device,
            parameters=parameters["_conv_stem"],
            input_params=[3, 2, 0, 32],
            use_shallow_covariant=True,
        )
        self._blocks0 = MBConvBlock(
            device,
            parameters["blocks"]["_blocks0"],
            is_depthwise_first=True,
            input_params=[[3, 1, 0, 8], [1, 1, 0, 8], [1, 1, 0, 32], [1, 1, 0, 16]],
            groups=32,
        )
        # self._blocks1 = MBConvBlock(
        #     device,
        #     parameters._blocks1,
        #     is_depthwise_first=False,
        #     input_params = [[3,1,0,8],[1,1,0,8],[1,1,0,32],[1,1,0,16] ]
        # )
        # self._blocks2 = MBConvBlock(
        #     device,
        #     parameters._blocks2,
        #     is_depthwise_first=False,
        # )
        # self._blocks3 = MBConvBlock(
        #     device,
        #     parameters._blocks3,
        #     is_depthwise_first=False,
        # )
        # self._blocks4 = MBConvBlock(
        #     device,
        #     parameters._blocks4,
        #     is_depthwise_first=False,
        # )
        # self._blocks5 = MBConvBlock(
        #     device,
        #     parameters._blocks5,
        #     is_depthwise_first=False,
        # )
        # self._blocks6 = MBConvBlock(
        #     device,
        #     parameters._blocks6,
        #     is_depthwise_first=False,
        # )
        # self._blocks7 = MBConvBlock(
        #     device,
        #     parameters._blocks7,
        #     is_depthwise_first=False,
        # )
        # self._blocks8 = MBConvBlock(
        #     device,
        #     parameters._blocks8,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._blocks9 = MBConvBlock(
        #     device,
        #     parameters._blocks9,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._blocks10 = MBConvBlock(
        #     device,
        #     parameters._blocks10,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._blocks11 = MBConvBlock(
        #     device,
        #     parameters._blocks11,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._blocks12 = MBConvBlock(
        #     device,
        #     parameters._blocks12,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._blocks13 = MBConvBlock(
        #     device,
        #     parameters._blocks13,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._blocks14 = MBConvBlock(
        #     device,
        #     parameters._blocks14,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._blocks15 = MBConvBlock(
        #     device,
        #     parameters._blocks15,
        #     is_depthwise_first=False,
        #     is_width_sharded=True,
        # )
        # self._conv_head = Conv2dDynamicSamePadding(
        #     device,
        #     parameters._conv_head,
        #     parameter_bn=parameters._bn1,
        #     conv_pth=parameters._conv_head.module,
        #     bn_pth=parameters._bn1.module,
        # )
        # self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        # self.l1_weight = parameters["l1"]["weight"]
        # self.l1_bias = parameters["l1"]["bias"]

    def __call__(self, x):
        x = self._conv_stem(x)
        x = ttnn.to_device(x, device=self.device)
        x = x * ttnn.sigmoid(x)
        torch.save(ttnn.to_torch(x), "models/experimental/functional_mobilenetv2/dumps/stemtt")
        x = self._blocks0(x)
        # x_1 = self._blocks1(x)
        # x = self._blocks2(x_1)
        # if x.is_sharded():
        #     x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # x = ttnn.add(x, x_1)
        # x_3 = self._blocks3(x)
        # x = self._blocks4(x_3)

        # if x.is_sharded():
        #     x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # if x_3.is_sharded():
        #     x_3 = ttnn.sharded_to_interleaved(x_3, ttnn.L1_MEMORY_CONFIG)

        # x = x + x_3
        # x_5 = self._blocks5(x)
        # x = self._blocks6(x_5)

        # if x_5.is_sharded():
        #     x_5 = ttnn.sharded_to_interleaved(x_5, ttnn.L1_MEMORY_CONFIG)

        # x_7_in = x + x_5
        # x = self._blocks7(x_7_in)

        # if x.is_sharded():
        #     x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # if x_7_in.is_sharded():
        #     x_7_in = ttnn.sharded_to_interleaved(x_7_in, ttnn.L1_MEMORY_CONFIG)

        # x = x_7_in + x
        # x_8 = self._blocks8(x)
        # print("block9")
        # x = self._blocks9(x_8)

        # # if(x.is_sharded()):
        # #     x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # # if(x_8.is_sharded()):
        # #     x_8 = ttnn.sharded_to_interleaved(x_8, ttnn.L1_MEMORY_CONFIG)

        # print("block10")
        # x_10_in = x + x_8
        # x = self._blocks10(x_10_in)
        # x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # x = x + x_10_in
        # x_11 = self._blocks11(x)
        # x = self._blocks12(x_11)

        # x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # x_11 = ttnn.sharded_to_interleaved(x_11, ttnn.L1_MEMORY_CONFIG)

        # x_13_in = x + x_11
        # x = self._blocks13(x_13_in)

        # x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # x_14_in = x + x_13_in
        # x = self._blocks14(x)

        # x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # x = x_14_in + x
        # x = self._blocks15(x)
        # x = self._conv_head(x)
        # x = x * ttnn.sigmoid(x)

        # x = input_preprocessing(
        #     x,
        #     self.batch,
        #     self.parameters._conv_head.out_channels,
        #     int(math.sqrt(x.shape[2])),
        #     int(math.sqrt(x.shape[2])),
        # )

        # x = self._avg_pooling(x)

        # x = torch.flatten(x, 1)

        # x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        # x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        # self.l1_weight = preprocess_linear_weight(self.l1_weight, dtype=ttnn.bfloat16)
        # self.l1_bias = preprocess_linear_bias(self.l1_bias, dtype=ttnn.bfloat16)
        # self.l1_weight = ttnn.to_device(self.l1_weight, self.device)
        # self.l1_bias = ttnn.to_device(self.l1_bias, self.device)

        # x = ttnn.linear(x, self.l1_weight, bias=self.l1_bias)

        return ttnn.from_device(x)
