# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
from tt_lib.utils import _nearest_y
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class Yolov9c_Conv2D:
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
        is_detect=False,
        is_dfl=False,
    ):
        self.is_detect = is_detect
        self.is_dfl = is_dfl
        self.conv = conv
        self.device = device
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
            packer_l1_acc=True,
            math_approx_mode=True,
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
            input_channels_alignment=32,
        )
        config_override = None
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
        if self.is_detect:
            input_height = int(math.sqrt(x.shape[2]))
            input_width = int(math.sqrt(x.shape[2]))
            batch_size = x.shape[0]
        elif self.is_dfl:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = self.conv.batch_size
            input_height = self.conv.input_height
            input_width = self.conv.input_width

        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
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


def Yolov9c_shard_SiLU(device, x, ncores=64):
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
    def __init__(self, device, parameter, conv_pt, enable_act=True, is_detect=False, enable_identity=False):
        self.enable_identity = enable_identity
        self.enable_act = enable_act
        self.conv = Yolov9c_Conv2D(parameter.conv, conv_pt.conv, device=device, is_detect=is_detect)

    def __call__(self, device, x):
        x = self.conv(x)

        if self.enable_act:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            if self.enable_identity:
                x = ttnn.identity(x, memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                x = ttnn.silu(x)

        return x


class RepConv:
    def __init__(self, device, parameter, conv_pt, is_detect=False, enable_identity=True):
        self.conv1 = Conv(device, parameter.conv1, conv_pt.conv1, enable_identity=enable_identity)
        self.conv2 = Conv(device, parameter.conv2, conv_pt.conv2, enable_identity=enable_identity)

    def __call__(self, device, x):
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        conv1_out = self.conv1(device, x)
        conv2_out = self.conv2(device, x)
        x = ttnn.silu(conv1_out + conv2_out)

        return x


class RepBottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = RepConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)

    def __call__(self, device, x):
        input = x
        x = self.cv1(device, x)
        x = self.cv2(device, x)
        return input + x


class RepCSP:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1, enable_identity=False)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2, enable_identity=False)
        self.cv3 = Conv(device, parameter.cv3, conv_pt.cv3, enable_identity=False)
        self.m = RepBottleneck(device, parameter.m[0], conv_pt.m[0])

    def __call__(self, device, x):
        cv1_out = self.cv1(device, x)
        m_out = self.m(device, cv1_out)
        cv2_out = self.cv2(device, x)
        if m_out.is_sharded():
            m_out = ttnn.sharded_to_interleaved(m_out, ttnn.L1_MEMORY_CONFIG)
        if cv2_out.is_sharded():
            cv2_out = ttnn.sharded_to_interleaved(cv2_out, ttnn.L1_MEMORY_CONFIG)

        concat_out = ttnn.concat((m_out, cv2_out), dim=-1)
        x = self.cv3(device, concat_out)
        ttnn.deallocate(cv1_out)
        ttnn.deallocate(m_out)
        ttnn.deallocate(cv2_out)
        ttnn.deallocate(concat_out)
        return x


class RepNCSPELAN4:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1, enable_identity=False)
        self.k1 = RepCSP(device, parameter.cv2[0], conv_pt.cv2[0])
        self.k2 = Conv(device, parameter.cv2[1], conv_pt.cv2[1], enable_identity=False)
        self.k3 = RepCSP(device, parameter.cv3[0], conv_pt.cv3[0])
        self.k4 = Conv(device, parameter.cv3[1], conv_pt.cv3[1], enable_identity=False)
        self.cv4 = Conv(device, parameter.cv4, conv_pt.cv4, enable_identity=False)

    def __call__(self, device, x):
        input = x
        x = self.cv1(device, x)
        y1, y2 = ttnn.split(x, 2, dim=-1)
        cv2_out = self.k1(device, y2)
        cv3_out = self.k2(device, cv2_out)
        cv4_out = self.k3(device, cv3_out)
        cv5_out = self.k4(device, cv4_out)
        cv4_out = ttnn.concat((y1, y2, cv3_out, cv5_out), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv4(device, cv4_out)
        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(cv2_out)
        ttnn.deallocate(cv3_out)
        ttnn.deallocate(cv4_out)
        return x


class ADown:
    def __init__(self, device, parameter, conv_pt):
        self.parameter = parameter
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1, enable_identity=False)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2, enable_identity=False)

    def __call__(self, device, x):
        input = x
        x = ttnn.to_device(x, device=device)
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[-1]))
        x = ttnn.permute(x, (0, 3, 1, 2))
        # x = ttnn.global_avg_pool2d(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_torch(x)
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x = ttnn.from_torch(
            x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]))

        x1 = ttnn.slice(x, slice_start=[0, 0, 0, 0], slice_end=[x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2])
        x2 = ttnn.slice(
            x, slice_start=[0, 0, 0, x.shape[3] // 2], slice_end=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]]
        )

        x1 = self.cv1(device, x1)
        if x2.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
        x2 = ttnn.max_pool2d(
            x2,
            batch_size=self.parameter.cv1.conv.batch_size,
            input_h=self.parameter.cv1.conv.input_height,
            input_w=self.parameter.cv1.conv.input_width,
            channels=self.parameter.cv1.conv.in_channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )
        x2 = self.cv2(device, x2)
        x = ttnn.concat((x1, x2), dim=-1)

        return x


class SPPELAN:
    def __init__(self, device, parameter, conv_pt):
        self.parameter = parameter
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1, enable_identity=False)
        self.cv5 = Conv(device, parameter.cv5, conv_pt.cv5, enable_identity=False)

    def __call__(self, device, x):
        x = ttnn.to_device(x, device=device)
        x1 = x
        m1 = ttnn.max_pool2d(
            x1,
            batch_size=self.parameter.cv2.batch_size,
            input_h=self.parameter.cv2.input_height,
            input_w=self.parameter.cv2.input_width,
            channels=self.parameter.cv1.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m2 = ttnn.max_pool2d(
            m1,
            batch_size=self.parameter.cv3.batch_size,
            input_h=self.parameter.cv3.input_height,
            input_w=self.parameter.cv3.input_width,
            channels=self.parameter.cv1.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m3 = ttnn.max_pool2d(
            m2,
            batch_size=self.parameter.cv4.batch_size,
            input_h=self.parameter.cv4.input_height,
            input_w=self.parameter.cv4.input_width,
            channels=self.parameter.cv1.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        if x1.is_sharded():
            x1 = ttnn.sharded_to_interleaved(x1, ttnn.L1_MEMORY_CONFIG)
        if m2.is_sharded():
            m2 = ttnn.sharded_to_interleaved(m2, ttnn.L1_MEMORY_CONFIG)
        if m3.is_sharded():
            m3 = ttnn.sharded_to_interleaved(m3, ttnn.L1_MEMORY_CONFIG)
        if m1.is_sharded():
            m1 = ttnn.sharded_to_interleaved(m1, ttnn.L1_MEMORY_CONFIG)
        y = ttnn.concat([x1, m1, m2, m3], dim=-1)

        return self.cv5(device, y)


class DFL:
    def __init__(self, device, parameter, conv_pt):
        self.dfl = Conv(device, parameter.dfl, conv_pt.dfl, enable_act=Fale, enable_identity=False)

    def __call__(self, device, x):
        return self.dfl(x)
