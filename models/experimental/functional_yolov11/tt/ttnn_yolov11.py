# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
from ttnn.model_preprocessing import ParameterDict, fold_batch_norm2d_into_conv2d
from torch import nn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
import math
import torch.nn.functional as F
from tt_lib.utils import (
    _nearest_y,
)
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


def input_preprocessing(x, N, C, H, W):
    x = ttnn.to_torch(x)
    x = torch.permute(x, (0, 3, 1, 2))
    x = x.reshape(N, C, H, W)
    return x


class Yolov11_Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
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
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = False
        self.cache = cache
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
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
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x


def Yolov11_shard_SiLU(device, x, ncores=64):
    input_2d_height = x.shape.with_tile_padding()[2]
    input_2d_width = x.shape.with_tile_padding()[3]

    input_2d_height_padded = _nearest_y(input_2d_height, ncores * 32)

    shard_height = math.ceil(input_2d_height_padded / ncores)
    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    shard_width = input_2d_width
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    tensor_memory_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)

    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

    x = ttnn.to_memory_config(x, memory_config=in_sharded_mem_config)

    x = ttnn.silu(x, memory_config=in_sharded_mem_config)
    return x


class Conv:
    def __init__(self, device, parameter, conv_pt, enable_act=True):
        self.enable_act = enable_act
        self.conv = Yolov11_Conv2D(parameter.conv, conv_pt.conv, device=device)

    def __call__(self, device, x):
        if self.enable_act:
            x = self.conv(x)
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.silu(x)

        else:
            x = self.conv(x)
        return x


class Bottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)

    def __call__(self, device, x):
        input = x
        x = self.cv1(device, x)
        x = self.cv2(device, x)
        return input + x


class SPPF:
    def __init__(self, device, parameter, conv_pt):
        self.parameter = parameter
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
        # self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def __call__(self, device, x):
        x = self.cv1(device, x)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x
        m1 = ttnn.max_pool2d(
            x,
            batch_size=self.parameter.cv2.conv.batch_size,
            input_h=self.parameter.cv2.conv.input_height,
            input_w=self.parameter.cv2.conv.input_width,
            channels=self.parameter.cv2.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            # applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        m2 = ttnn.max_pool2d(
            m1,
            batch_size=self.parameter.cv2.conv.batch_size,
            input_h=self.parameter.cv2.conv.input_height,
            input_w=self.parameter.cv2.conv.input_width,
            channels=self.parameter.cv2.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            # applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        m3 = ttnn.max_pool2d(
            m2,
            batch_size=self.parameter.cv2.conv.batch_size,
            input_h=self.parameter.cv2.conv.input_height,
            input_w=self.parameter.cv2.conv.input_width,
            channels=self.parameter.cv2.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            # applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        if x1.is_sharded():
            x1 = ttnn.sharded_to_interleaved(x1, ttnn.DRAM_MEMORY_CONFIG)
        if m2.is_sharded():
            m2 = ttnn.sharded_to_interleaved(m2, ttnn.DRAM_MEMORY_CONFIG)
        if m3.is_sharded():
            m3 = ttnn.sharded_to_interleaved(m3, ttnn.DRAM_MEMORY_CONFIG)
        if m1.is_sharded():
            m1 = ttnn.sharded_to_interleaved(m1, ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.concat([x1, m1, m2, m3], dim=-1)

        x = self.cv2(device, y)

        x = x[:, :, :49, :]
        return x


class C3K:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
        self.cv3 = Conv(device, parameter.cv3, conv_pt.cv3)
        self.k1 = Bottleneck(device, parameter.m[0], conv_pt.m[0])
        self.k2 = Bottleneck(device, parameter.m[1], conv_pt.m[1])

    def __call__(self, device, x):
        x1 = self.cv1(device, x)
        x2 = self.cv2(device, x)

        k1 = self.k1(device, x1)
        k2 = self.k2(device, k1)

        if x2.is_sharded():
            x2 = ttnn.sharded_to_interleaved(x2, ttnn.L1_MEMORY_CONFIG)
        if k2.is_sharded():
            k2 = ttnn.sharded_to_interleaved(k2, ttnn.L1_MEMORY_CONFIG)

        x = ttnn.concat((k2, x2), 3)
        x = self.cv3(device, x)
        return x


class C3k2:
    def __init__(self, device, parameter, conv_pt, is_bk_enabled=False):
        self.is_bk_enabled = is_bk_enabled
        self.parameter = parameter

        if is_bk_enabled:
            self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
            self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
            self.k = Bottleneck(device, parameter[0], conv_pt.m[0])
        else:
            self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
            self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
            self.c3k = C3K(device, parameter[0], conv_pt.m[0])

    def __call__(self, device, x):
        if self.is_bk_enabled:
            x = self.cv1(device, x)

            # if x.is_sharded():
            #     x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

            x = ttnn.to_torch(x)
            y1, y2 = x.chunk(2, -1)

            # y1, y2 = ttnn.split(x, 2, 3)
            # print(y1.shape, y2.shape)

            y1 = ttnn.from_torch(y1, dtype=ttnn.bfloat16, device=device)

            y2 = ttnn.from_torch(y2, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            y3 = self.k(device, y2)

            y2 = ttnn.to_layout(y2, ttnn.ROW_MAJOR_LAYOUT)
            y3 = ttnn.to_layout(y3, ttnn.ROW_MAJOR_LAYOUT)

            x = ttnn.concat((y1, y2, y3), 3)

            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = self.cv2(device, x)
        else:
            x = self.cv1(device, x)

            x = ttnn.to_torch(x)
            y1, y2 = x.chunk(2, -1)

            y1 = ttnn.from_torch(y1, dtype=ttnn.bfloat16, device=device)

            y2 = ttnn.from_torch(y2, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            y3 = self.c3k(device, y2)

            if y1.is_sharded():
                y1 = ttnn.sharded_to_interleaved(y1, ttnn.L1_MEMORY_CONFIG)
            if y2.is_sharded():
                y2 = ttnn.sharded_to_interleaved(y2, ttnn.L1_MEMORY_CONFIG)
            if y3.is_sharded():
                y3 = ttnn.sharded_to_interleaved(y3, ttnn.L1_MEMORY_CONFIG)

            y2 = ttnn.to_layout(y2, ttnn.ROW_MAJOR_LAYOUT)
            y3 = ttnn.to_layout(y3, ttnn.ROW_MAJOR_LAYOUT)

            x = ttnn.concat((y1, y2, y3), 3)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = self.cv2(device, x)
        return x


class Attention:
    def __init__(self, device, parameter, conv_pt):
        # print(parameter,conv)
        self.qkv = Conv(device, parameter.qkv, conv_pt.qkv, enable_act=False)
        self.proj = Conv(device, parameter.proj, conv_pt.proj, enable_act=False)
        self.pe = Conv(device, parameter.pe, conv_pt.pe, enable_act=False)
        self.num_heads = 2
        self.key_dim = 32
        self.head_dim = 64
        self.scale = self.key_dim**-0.5

    def __call__(self, device, x, batch_size=1):
        qkv = self.qkv(device, x)  # [1, 1, 49[64], 256]
        qkv = ttnn.sharded_to_interleaved(qkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))  # [1,256,1,49]
        qkv = ttnn.reshape(
            qkv, (batch_size, self.num_heads, self.key_dim * 2 + self.head_dim, qkv.shape[-1])
        )  # [1,2,128,49]
        q, k, v = (
            qkv[:, :, : self.key_dim, :],
            qkv[:, :, self.key_dim : self.head_dim, :],
            qkv[:, :, self.head_dim :, :],
        )  # ttnn.Shape([1, 2, 32, 49[64]]) ttnn.Shape([1, 2, 32, 49[64]]) ttnn.Shape([1, 2, 64, 49[64]])
        # print("------",q.shape,k.shape,v.shape)
        q_permuted = ttnn.permute(q, (0, 1, 3, 2))  # ttnn.Shape([1, 2, 49[64]],32)
        attn = ttnn.matmul(q_permuted, k)
        attn = ttnn.multiply(attn, self.scale)  # ([1, 2, 49, 49])
        attn = ttnn.softmax(attn, dim=-1)
        attn = ttnn.permute(attn, (0, 1, 3, 2))
        x1 = ttnn.matmul(v, attn)  # [1, 2, 64, 49[64]]
        x1 = ttnn.reshape(x1, (1, 1, (x1.shape[0] * x1.shape[1] * x1.shape[2]), x1.shape[3]))
        x1 = ttnn.permute(x1, (0, 1, 3, 2))
        v = ttnn.reshape(v, (1, 1, (v.shape[0] * v.shape[1] * v.shape[2]), v.shape[3]))  # [1,1,128, 49[64]]
        v = ttnn.permute(v, (0, 1, 3, 2))
        x2 = self.pe(device=device, x=v)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        # print("shape before add", x1.shape, x2.shape)
        x = x1 + x2
        # print("------", x.shape)
        x = self.proj(device=device, x=x)
        return x


class PSABlock:
    def __init__(self, device, parameter, conv_pt):
        self.attn = Attention(device=device, parameter=parameter.attn, conv_pt=conv_pt.attn)
        self.ffn_conv1 = Conv(device, parameter.ffn[0], conv_pt.ffn[0])
        self.ffn_conv2 = Conv(device, parameter.ffn[1], conv_pt.ffn[1], enable_act=False)

    def __call__(self, device, x):
        x1 = x
        x = self.attn(device, x)
        x = x1 + x
        x1 = x
        x = self.ffn_conv1(device, x)
        x = self.ffn_conv2(device, x)
        return x + x1


class C2PSA:
    def __init__(self, device, parameter, conv_pt):
        self.out_channel_0 = parameter.cv1.conv.out_channels
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
        self.psablock = PSABlock(device, parameter.psablock, conv_pt.psablock)

    def __call__(self, device, x):
        x = self.cv1(device, x)  # (1,1,49,256)
        a, b = x[:, :, :, : int(self.out_channel_0 / 2)], x[:, :, :, int(self.out_channel_0 / 2) :]
        x = self.psablock(device, b)
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.concat((a, x), dim=-1)
        x = self.cv2(device, x)
        return x


class YoloV11:
    def __init__(self, device, parameters):
        self.device = device
        # print(parameters.model[0])
        self.conv1 = Conv(device, parameters.conv_args[0], parameters.model[0])
        self.conv2 = Conv(device, parameters.conv_args[1], parameters.model[1])
        self.c3k2_1 = C3k2(device, parameters.conv_args[2], parameters.model[2], is_bk_enabled=True)
        self.conv3 = Conv(device, parameters.conv_args[3], parameters.model[3])
        self.c3k2_2 = C3k2(device, parameters.conv_args[4], parameters.model[4], is_bk_enabled=True)
        self.conv5 = Conv(device, parameters.conv_args[5], parameters.model[5])
        self.c3k2_3 = C3k2(device, parameters.conv_args[6], parameters.model[6], is_bk_enabled=False)
        # self.conv6 = Conv(device, parameters.conv_args[7], parameters.model[7])
        # self.c3k2_4 = C3k2(device, parameters.conv_args[8], parameters.model[8], is_bk_enabled=True)

    def __call__(self, x):
        x = self.conv1(self.device, x)
        x = self.conv2(self.device, x)
        x = self.c3k2_1(self.device, x)
        x = self.conv3(self.device, x)  # 3 #0.998
        x = self.c3k2_2(self.device, x)  # 4
        x4 = x
        x = self.conv5(self.device, x)  # 5
        x = self.c3k2_3(self.device, x)  # 6
        return x
        x6 = x
        x = self.conv6(self.device, x)  # 7
        x = self.c3k2_4(self.device, x)  # 8

        return x
