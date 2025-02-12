# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
import math
from tt_lib.utils import (
    _nearest_y,
)
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import torch


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
            fp32_dest_acc_en=False,
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


def Yolov11_shard_upsample(device, x):
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 5),
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 6),
                ttnn.CoreCoord(0, 6),
            ),
        }
    )
    shard_height = math.ceil(x.shape[0] * x.shape[1] * x.shape[2] / 49)
    shard_width = x.shape[-1]
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    in_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    x = ttnn.to_memory_config(x, memory_config=in_sharded_mem_config)
    shard_height_out = shard_height * 2 * 2  # scale_factor=2
    shard_spec_out = ttnn.ShardSpec(shard_grid, (shard_height_out, shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    out_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_out
    )
    x = ttnn.upsample(x, scale_factor=2, memory_config=out_sharded_mem_config)
    x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
    return x


class Conv:
    def __init__(self, device, parameter, conv_pt, enable_act=True, is_detect=False):
        self.enable_act = enable_act
        self.conv = Yolov11_Conv2D(parameter.conv, conv_pt.conv, device=device, is_detect=is_detect)

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

    def __call__(self, device, x):
        x = self.cv1(device, x)
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
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
        )
        use_sharded_concat = True
        if use_sharded_concat:
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            if m1.shape[2] == 49:  # 224
                shard_height = 1  # (n*h*w + num_cores - 1)//num_cores
            elif m1.shape[2] == 400:  # 640
                shard_height = 7
            else:
                print("invalid shard spec")
            in_shard_width = x1.shape[-1]
            out_shard_width = x1.shape[-1] + m1.shape[-1] + m2.shape[-1] + m3.shape[-1]
            input_sharded_memory_config = ttnn.create_sharded_memory_config(
                (shard_height, in_shard_width),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            output_sharded_memory_config = ttnn.create_sharded_memory_config(
                (shard_height, out_shard_width),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            print("inpit is", input_sharded_memory_config)
            print("outpt is", output_sharded_memory_config)
            x1 = ttnn.to_memory_config(x1, memory_config=input_sharded_memory_config)
            m1 = ttnn.to_memory_config(m1, memory_config=input_sharded_memory_config)
            m2 = ttnn.to_memory_config(m2, memory_config=input_sharded_memory_config)
            m3 = ttnn.to_memory_config(m3, memory_config=input_sharded_memory_config)
            memory_config_used = output_sharded_memory_config
        else:
            if m2.is_sharded():
                m2 = ttnn.sharded_to_interleaved(m2, ttnn.L1_MEMORY_CONFIG)
            if m3.is_sharded():
                m3 = ttnn.sharded_to_interleaved(m3, ttnn.L1_MEMORY_CONFIG)
            if m1.is_sharded():
                m1 = ttnn.sharded_to_interleaved(m1, ttnn.L1_MEMORY_CONFIG)
            memory_config_used = ttnn.L1_MEMORY_CONFIG
        y = ttnn.concat([x1, m1, m2, m3], dim=-1, memory_config=memory_config_used)
        if y.is_sharded():
            y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(device, y)
        ttnn.deallocate(x1)
        ttnn.deallocate(m1)
        ttnn.deallocate(m2)
        ttnn.deallocate(m3)
        x = ttnn.reallocate(x)
        print("output shape is ", x.shape)
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
        print(
            "input to concat",
            x2.shape,
            x2.layout,
            x2.dtype,
            x2.memory_config(),
            k2.shape,
            k2.layout,
            k2.dtype,
            k2.memory_config(),
        )
        use_shard_concat = False  # fps drop due to layout conversion
        if use_shard_concat:
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            if x2.shape[2] == 49:  # 224
                shard_height = 1  # (n*h*w + num_cores - 1)//num_cores
            elif x2.shape[2] == 196:  # 224
                shard_height = 4
            elif x2.shape[2] == 400:  # 640
                shard_height = 7
            elif x2.shape[2] == 1600:  # 640
                shard_height = 13
            else:
                print("invalid shard spec")
            in_shard_width = x2.shape[-1]
            out_shard_width = x2.shape[-1] + k2.shape[-1]
            input_sharded_memory_config = ttnn.create_sharded_memory_config(
                (shard_height, in_shard_width),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            x2 = ttnn.to_dtype(x2, ttnn.bfloat16)
            x2 = ttnn.to_memory_config(x2, memory_config=input_sharded_memory_config)
            k2 = ttnn.to_layout(k2, ttnn.ROW_MAJOR_LAYOUT)
            k2 = ttnn.to_dtype(k2, ttnn.bfloat16)
            k2 = ttnn.to_memory_config(k2, memory_config=input_sharded_memory_config)
            output_sharded_memory_config = ttnn.create_sharded_memory_config(
                (shard_height, out_shard_width),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            memory_config_used = output_sharded_memory_config
        else:
            memory_config_used = ttnn.L1_MEMORY_CONFIG
        x = ttnn.concat((k2, x2), 3, memory_config=memory_config_used)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv3(device, x)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(k1)
        ttnn.deallocate(k2)
        x = ttnn.reallocate(x)
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
        # if self.is_bk_enabled:
        x = self.cv1(device, x)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        print("x.sha[e is ]", x.shape)
        y1 = x[:, :, :, : x.shape[-1] // 2]
        y2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        # x = ttnn.reallocate(x)
        # y2 = ttnn.to_layout(y2, layout=ttnn.TILE_LAYOUT)
        if self.is_bk_enabled:
            y2 = ttnn.to_layout(y2, layout=ttnn.TILE_LAYOUT)
            y3 = self.k(device, y2)
        else:
            y3 = self.c3k(device, y2)

        if y2.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            y2 = ttnn.to_layout(y2, ttnn.ROW_MAJOR_LAYOUT)
        if y3.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            y3 = ttnn.to_layout(y3, ttnn.ROW_MAJOR_LAYOUT)

        print("y1,y2,y3", y1.shape, y2.shape, y3.shape, y1.dtype, y2.dtype, y3.dtype)
        use_shard_concat = True
        if use_shard_concat:
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            if y1.shape[2] == 25600:  # 640
                shard_height = 400  # (n*h*w + num_cores - 1)//num_cores
            elif y1.shape[2] == 6400:  # 640
                shard_height = 102
            elif y1.shape[2] == 1600:  # 640
                shard_height = 25
            elif y1.shape[2] == 400:  # 640
                shard_height = 7
            elif y1.shape[2] == 3136:  # 224
                shard_height = 49
            elif y1.shape[2] == 784:  # 224
                shard_height = 13
            elif y1.shape[2] == 196:  # 224
                shard_height = 4
            elif y1.shape[2] == 49:  # 224
                shard_height = 1
            else:
                print("invalid shard spec")
            in_shard_width = y1.shape[-1]
            out_shard_width = y1.shape[-1] + y2.shape[-1] + y3.shape[-1]
            input_sharded_memory_config = ttnn.create_sharded_memory_config(
                (shard_height, in_shard_width),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            output_sharded_memory_config = ttnn.create_sharded_memory_config(
                (shard_height, out_shard_width),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            y1 = ttnn.to_memory_config(y1, memory_config=input_sharded_memory_config)
            y2 = ttnn.to_memory_config(y2, memory_config=input_sharded_memory_config)
            y3 = ttnn.to_memory_config(y3, memory_config=input_sharded_memory_config)
            memory_config_used = output_sharded_memory_config
        else:
            memory_config_used = ttnn.L1_MEMORY_CONFIG
        x = ttnn.concat((y1, y2, y3), 3, memory_config=memory_config_used)

        if use_shard_concat:
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        # if x.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
        #     x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        x = self.cv2(device, x)

        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)
        x = ttnn.reallocate(x)
        return x


class Attention:
    def __init__(self, device, parameter, conv_pt):
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
        print("before qkv details", qkv.shape, qkv.layout, qkv.dtype)
        # qkv = ttnn.to_torch(qkv)
        # qkv = ttnn.from_torch(
        #     qkv, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        # )
        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT)
        qkv = ttnn.to_dtype(qkv, ttnn.bfloat16)
        qkv = ttnn.to_layout(qkv, layout=ttnn.TILE_LAYOUT)
        print("after qkv details", qkv.shape, qkv.layout, qkv.dtype)
        qkv = ttnn.reshape(
            qkv, (batch_size, self.num_heads, self.key_dim * 2 + self.head_dim, qkv.shape[-1])
        )  # [1,2,128,49]
        q, k, v = (
            qkv[:, :, : self.key_dim, :],
            qkv[:, :, self.key_dim : self.head_dim, :],
            qkv[:, :, self.head_dim :, :],
        )  # ttnn.Shape([1, 2, 32, 49[64]]) ttnn.Shape([1, 2, 32, 49[64]]) ttnn.Shape([1, 2, 64, 49[64]])

        q_permuted = ttnn.permute(q, (0, 1, 3, 2))  # ttnn.Shape([1, 2, 49[64]],32)
        attn = ttnn.matmul(q_permuted, k, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.multiply(attn, self.scale)  # ([1, 2, 49, 49])
        attn = ttnn.softmax(attn, dim=-1)
        attn = ttnn.permute(attn, (0, 1, 3, 2))
        x1 = ttnn.matmul(v, attn, memory_config=ttnn.L1_MEMORY_CONFIG)  # [1, 2, 64, 49[64]]
        x1 = ttnn.reshape(x1, (1, 1, (x1.shape[0] * x1.shape[1] * x1.shape[2]), x1.shape[3]))
        x1 = ttnn.permute(x1, (0, 1, 3, 2))
        v = ttnn.reshape(v, (1, 1, (v.shape[0] * v.shape[1] * v.shape[2]), v.shape[3]))  # [1,1,128, 49[64]]
        v = ttnn.permute(v, (0, 1, 3, 2))
        x2 = self.pe(device=device, x=v)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = x1 + x2

        x = self.proj(device=device, x=x)
        ttnn.deallocate(x1)
        ttnn.deallocate(qkv)
        ttnn.deallocate(q_permuted)
        ttnn.deallocate(attn)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(x2)
        x = ttnn.reallocate(x)
        return x


def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=64) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int = 8, grid_cols: int = 8):
    rows = num_cores // grid_cols
    assert rows <= grid_rows, "Not enough cores for specified core grid"
    ranges = []
    if rows != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, rows - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, rows),
                ttnn.CoreCoord(remainder - 1, rows),
            )
        )
    return ttnn.CoreRangeSet({*ranges})


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
        self.psablock = PSABlock(device, parameter.m[0], conv_pt.m[0])

    def __call__(self, device, x):
        x = self.cv1(device, x)  # (1,1,49,256)
        a, b = x[:, :, :, : int(self.out_channel_0 / 2)], x[:, :, :, int(self.out_channel_0 / 2) :]
        x = self.psablock(device, b)
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.concat((a, x), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(device, x)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        x = ttnn.reallocate(x)
        return x


class Detect:
    def __init__(self, device, parameter, conv_pt):
        self.cv2_0_0 = Conv(device, parameter.cv2[0][0], conv_pt.cv2[0][0], is_detect=True)
        self.cv2_0_1 = Conv(device, parameter.cv2[0][1], conv_pt.cv2[0][1], is_detect=True)
        self.cv2_0_2 = Yolov11_Conv2D(parameter.cv2[0][2], conv_pt.cv2[0][2], device=device, is_detect=True)

        self.cv2_1_0 = Conv(device, parameter.cv2[1][0], conv_pt.cv2[1][0], is_detect=True)
        self.cv2_1_1 = Conv(device, parameter.cv2[1][1], conv_pt.cv2[1][1], is_detect=True)
        self.cv2_1_2 = Yolov11_Conv2D(parameter.cv2[1][2], conv_pt.cv2[1][2], device=device, is_detect=True)

        self.cv2_2_0 = Conv(device, parameter.cv2[2][0], conv_pt.cv2[2][0], is_detect=True)
        self.cv2_2_1 = Conv(device, parameter.cv2[2][1], conv_pt.cv2[2][1], is_detect=True)
        self.cv2_2_2 = Yolov11_Conv2D(parameter.cv2[2][2], conv_pt.cv2[2][2], device=device, is_detect=True)

        self.cv3_0_0_0 = Conv(device, parameter.cv3[0][0][0], conv_pt.cv3[0][0][0], is_detect=True)
        self.cv3_0_0_1 = Conv(device, parameter.cv3[0][0][1], conv_pt.cv3[0][0][1], is_detect=True)
        self.cv3_0_1_0 = Conv(device, parameter.cv3[0][1][0], conv_pt.cv3[0][1][0], is_detect=True)
        self.cv3_0_1_1 = Conv(device, parameter.cv3[0][1][1], conv_pt.cv3[0][1][1], is_detect=True)
        self.cv3_0_2_0 = Yolov11_Conv2D(parameter.cv3[0][2], conv_pt.cv3[0][2], device=device, is_detect=True)

        self.cv3_1_0_0 = Conv(device, parameter.cv3[1][0][0], conv_pt.cv3[1][0][0], is_detect=True)
        self.cv3_1_0_1 = Conv(device, parameter.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True)
        self.cv3_1_1_0 = Conv(device, parameter.cv3[1][1][0], conv_pt.cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = Conv(device, parameter.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True)
        self.cv3_1_2_0 = Yolov11_Conv2D(parameter.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True)

        self.cv3_2_0_0 = Conv(device, parameter.cv3[2][0][0], conv_pt.cv3[2][0][0], is_detect=True)
        self.cv3_2_0_1 = Conv(device, parameter.cv3[2][0][1], conv_pt.cv3[2][0][1], is_detect=True)
        self.cv3_2_1_0 = Conv(device, parameter.cv3[2][1][0], conv_pt.cv3[2][1][0], is_detect=True)
        self.cv3_2_1_1 = Conv(device, parameter.cv3[2][1][1], conv_pt.cv3[2][1][1], is_detect=True)
        self.cv3_2_2_0 = Yolov11_Conv2D(parameter.cv3[2][2], conv_pt.cv3[2][2], device=device, is_detect=True)

        self.dfl = Yolov11_Conv2D(parameter.dfl.conv, conv_pt.dfl.conv, device=device, is_dfl=True)
        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, device, y1, y2, y3):  # 0.9934, #0.987, #0.9803
        x1 = self.cv2_0_0(device, y1)
        x1 = self.cv2_0_1(device, x1)
        x1 = self.cv2_0_2(x1)  # 0.98
        x2 = self.cv2_1_0(device, y2)
        x2 = self.cv2_1_1(device, x2)
        x2 = self.cv2_1_2(x2)  # 0.993

        x3 = self.cv2_2_0(device, y3)
        x3 = self.cv2_2_1(device, x3)
        x3 = self.cv2_2_2(x3)  # 0.998

        x4 = self.cv3_0_0_0(device, y1)
        x4 = self.cv3_0_0_1(device, x4)
        x4 = self.cv3_0_1_0(device, x4)
        x4 = self.cv3_0_1_1(device, x4)
        x4 = self.cv3_0_2_0(x4)  # 0.986

        x5 = self.cv3_1_0_0(device, y2)
        x5 = self.cv3_1_0_1(device, x5)
        x5 = self.cv3_1_1_0(device, x5)
        x5 = self.cv3_1_1_1(device, x5)
        x5 = self.cv3_1_2_0(x5)  # 0.961

        x6 = self.cv3_2_0_0(device, y3)
        x6 = self.cv3_2_0_1(device, x6)
        x6 = self.cv3_2_1_0(device, x6)
        x6 = self.cv3_2_1_1(device, x6)
        x6 = self.cv3_2_2_0(x6)  # 0.986

        x1 = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x4 = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        x5 = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1 = ttnn.concat((x1, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.concat((x2, x5), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3 = ttnn.concat((x3, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1_reshaped = ttnn.reshape(y1, (y1.shape[0], y1.shape[2], y1.shape[-1]))  # 0.9992
        y2_reshaped = ttnn.reshape(y2, (y2.shape[0], y2.shape[2], y2.shape[-1]))  # 0.99908
        y3_reshaped = ttnn.reshape(y3, (y3.shape[0], y3.shape[2], y3.shape[-1]))  # 0.993

        # y_all = [y1_reshaped, y2_reshaped, y3_reshaped]
        y = ttnn.concat((y1_reshaped, y2_reshaped, y3_reshaped), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.998
        ya, yb = y[:, :, :64], y[:, :, 64:144]  # 0.991, 0.97
        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)
        ttnn.deallocate(x4)
        ttnn.deallocate(x5)
        ttnn.deallocate(x6)
        ttnn.deallocate(y1_reshaped)
        ttnn.deallocate(y2_reshaped)
        ttnn.deallocate(y3_reshaped)
        ttnn.deallocate(y)
        ya = ttnn.reallocate(ya)
        yb = ttnn.reallocate(yb)
        ya = ttnn.permute(ya, (0, 2, 1))
        print("before reshape", ya.shape, ya.layout, ya.memory_config(), ya.dtype)
        ya = ttnn.reshape(ya, (ya.shape[0], 4, 16, ya.shape[2]))
        ya = ttnn.permute(ya, (0, 2, 1, 3))  # 0.991
        ya = ttnn.to_layout(ya, ttnn.TILE_LAYOUT)
        ya = ttnn.softmax(ya, dim=1)
        ya = ttnn.permute(ya, (0, 2, 3, 1))
        c = self.dfl(ya)  # 0.968
        ttnn.deallocate(ya)
        c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)

        c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]

        anchor, strides = self.anchors, self.strides
        anchor = ttnn.to_memory_config(anchor, memory_config=ttnn.L1_MEMORY_CONFIG)
        strides = ttnn.to_memory_config(strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)

        c1 = anchor - c1  # 0.998
        c2 = anchor + c2  # 0.997

        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)  # 0.9995

        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        z = ttnn.multiply(z, strides)  # 0.9998
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)
        ttnn.deallocate(c)
        ttnn.deallocate(z1)
        ttnn.deallocate(z2)
        ttnn.deallocate(c1)
        ttnn.deallocate(c2)
        ttnn.deallocate(anchor)
        ttnn.deallocate(strides)
        z = ttnn.reallocate(z)
        yb = ttnn.reallocate(yb)
        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        ttnn.deallocate(yb)

        ttnn.deallocate(z)

        # out = ttnn.reallocate(out)
        return out


class YoloV11:
    def __init__(self, device, parameters):
        self.device = device

        self.conv1 = Conv(device, parameters.conv_args[0], parameters.model[0])
        self.conv2 = Conv(device, parameters.conv_args[1], parameters.model[1])
        self.c3k2_1 = C3k2(device, parameters.conv_args[2], parameters.model[2], is_bk_enabled=True)
        self.conv3 = Conv(device, parameters.conv_args[3], parameters.model[3])
        self.c3k2_2 = C3k2(device, parameters.conv_args[4], parameters.model[4], is_bk_enabled=True)
        self.conv5 = Conv(device, parameters.conv_args[5], parameters.model[5])
        self.c3k2_3 = C3k2(device, parameters.conv_args[6], parameters.model[6], is_bk_enabled=False)
        self.conv6 = Conv(device, parameters.conv_args[7], parameters.model[7])
        self.c3k2_4 = C3k2(device, parameters.conv_args[8], parameters.model[8], is_bk_enabled=False)
        self.sppf = SPPF(device, parameters.conv_args[9], parameters.model[9])
        self.c2psa = C2PSA(device, parameters.conv_args[10], parameters.model[10])
        self.c3k2_5 = C3k2(
            device,
            parameters.conv_args[13],
            parameters.model[13],
            is_bk_enabled=True,
        )
        self.c3k2_6 = C3k2(
            device,
            parameters.conv_args[16],
            parameters.model[16],
            is_bk_enabled=True,
        )
        self.conv7 = Conv(device, parameters.conv_args[17], parameters.model[17])
        self.c3k2_7 = C3k2(
            device,
            parameters.conv_args[19],
            parameters.model[19],
            is_bk_enabled=True,
        )
        self.conv8 = Conv(device, parameters.conv_args[20], parameters.model[20])
        self.c3k2_8 = C3k2(
            device,
            parameters.conv_args[22],
            parameters.model[22],
            is_bk_enabled=False,
        )
        self.detect = Detect(device, parameters.model_args.model[23], parameters.model[23])

    def __call__(self, x):
        x = self.conv1(self.device, x)  # 0.9997768588395117
        x = self.conv2(self.device, x)  # 0.9995776757085278
        x = self.c3k2_1(self.device, x)  # 0.9981333502789589
        x = self.conv3(self.device, x)  # 0.9969896145281048
        x = self.c3k2_2(self.device, x)  # -.0.07, WITH FROM AND TO - 0.9947373339421013
        # torch.save(ttnn.to_torch(x).reshape(x.shape[0],int(torch.sqrt(torch.tensor(x.shape[2], dtype=torch.float32))),int(torch.sqrt(torch.tensor(x.shape[2], dtype=torch.float32))),x.shape[-1]).permute(0,3,1,2),"/home/ubuntu/tt-metal/models/experimental/functional_yolov11/dumps/ttnn_out.pth")
        # return x
        x4 = x
        x = self.conv5(self.device, x)  # 0.9958605201742154
        x = self.c3k2_3(self.device, x)  # 6

        x6 = x
        x = self.conv6(self.device, x)  # 7
        x = self.c3k2_4(self.device, x)  # 8 #0.25

        x = self.sppf(self.device, x)  # 9 #0.986
        x = self.c2psa(self.device, x)  # 10
        print("before x details", x.shape, x.layout, x.dtype)
        # x = ttnn.to_torch(x)
        # x = ttnn.from_torch(
        #     x,
        #     dtype=ttnn.bfloat16,
        #     device=self.device,
        #     layout=ttnn.ROW_MAJOR_LAYOUT,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        # )
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_dtype(x, ttnn.bfloat16)
        # torch.save(ttnn.to_torch(x).reshape(1,7,7,256).permute(0,3,1,2),"/home/ubuntu/venkatesh_yolov11/tt-metal/models/experimental/functional_yolov11/dumps/tt_out.pth")
        print("after x details", x.shape, x.layout, x.dtype)
        x10 = x
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
        print("ttnn input to upsample1 is ", x.shape, x.layout, x.dtype)
        # x = Yolov11_shard_upsample(self.device, x)
        # x = ttnn.upsample(x, scale_factor=2)
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores_for_upsample(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)

        x = ttnn.upsample(x, scale_factor=2, memory_config=x.memory_config())  # 11
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        print("output of 1st upsample", x.shape)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        x6 = ttnn.to_layout(x6, layout=ttnn.ROW_MAJOR_LAYOUT)
        print("x and x6 and x4 oconfig is ", x.memory_config(), x6.memory_config(), x4.memory_config())
        x = ttnn.concat((x, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 12
        ttnn.deallocate(x6)
        # x = ttnn.reallocate(x)
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        x = self.c3k2_5(self.device, x)  # 13
        x13 = x
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
        print("ttnn input to upsample2 is ", x.shape, x.layout, x.dtype)
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores_for_upsample(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)
        x = ttnn.upsample(x, scale_factor=2, memory_config=x.memory_config())
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        # x = ttnn.upsample(x, scale_factor=2)  # 14
        # x = Yolov11_shard_upsample(self.device, x)
        print("output of 2nd upsample", x.shape)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        x4 = ttnn.to_layout(x4, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.concat((x, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 15
        ttnn.deallocate(x4)
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        x = self.c3k2_6(self.device, x)  # 16
        x16 = x
        x = self.conv7(self.device, x)  # 17
        # x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        # x = ttnn.to_dtype(x, ttnn.bfloat16)
        # x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        print("x and x13 shapes are", x.shape, x13.shape, x.dtype, x13.dtype, x.layout, x13.layout)
        x = ttnn.concat((x, x13), -1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 18
        ttnn.deallocate(x13)
        x = self.c3k2_7(self.device, x)  # 19
        x19 = x
        x = self.conv8(self.device, x)  # 20 #16
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_dtype(x, ttnn.bfloat16)
        print("x and x10 shapes are", x.shape, x10.shape, x.dtype, x10.dtype, x.layout, x10.layout)
        x = ttnn.concat((x, x10), -1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 21
        print("output cncat shape is", x.shape)
        ttnn.deallocate(x10)
        x = self.c3k2_8(self.device, x)  # 22
        x22 = x
        x = self.detect(self.device, x16, x19, x22)
        # ttnn.deallocate(x16)
        # ttnn.deallocate(x19)
        # ttnn.deallocate(x22)
        return x
