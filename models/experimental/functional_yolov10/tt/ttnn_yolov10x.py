# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.experimental.yolo_common.yolo_utils import determine_num_cores, get_core_grid_from_num_cores


def interleaved_to_sharded(x):
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
    nhw = x.shape[0] * x.shape[1] * x.shape[2]
    num_cores = determine_num_cores(nhw, x.shape[2])
    core_grid = get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    return ttnn.reshard(x, shardspec) if x.is_sharded() else ttnn.interleaved_to_sharded(x, shardspec)


class TtYolov10_Conv2D:
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
        config_override=None,
        auto_shard=False,
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
        self.auto_shard = auto_shard
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        if shard_layout is None and not self.auto_shard:
            shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if use_1d_systolic_array
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            input_channels_alignment=32,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
            enable_subblock_padding=False,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if auto_shard:
            self.conv_config.shard_layout = None

        config_override = None
        config_override = {"act_block_h": 64} if conv.in_channels == 3 else None
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

        [x, [output_height, output_width]] = ttnn.conv2d(
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
            return_weights_and_bias=False,
        )
        return x


class Conv:
    def __init__(
        self,
        device,
        parameter,
        conv_pt,
        enable_act=True,
        is_detect=False,
        enable_identity=False,
        is_dfl=False,
        config_override=None,
        use_1d_systolic_array=True,
        auto_shard=False,
        shard_layout=None,
        activation="",
    ):
        self.enable_identity = enable_identity
        self.enable_act = enable_act
        if not self.enable_identity:
            activation = "silu"
        self.conv = TtYolov10_Conv2D(
            parameter.conv,
            conv_pt.conv,
            device=device,
            is_detect=is_detect,
            is_dfl=is_dfl,
            config_override=config_override,
            use_1d_systolic_array=use_1d_systolic_array,
            auto_shard=auto_shard,
            shard_layout=shard_layout,
            activation=activation,
        )

    def __call__(self, x):
        x = self.conv(x)

        if self.enable_act:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            if self.enable_identity:
                x = ttnn.identity(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        return x


class TtnnBottleNeck:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
            auto_shard=True,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            auto_shard=True,
        )

    def __call__(self, x):
        cv1 = self.cv1(x)
        cv2 = self.cv2(cv1)
        ttnn.deallocate(cv1)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv2


class TtnnSCDown:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
            auto_shard=True,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            enable_identity=True,
            use_1d_systolic_array=False,
            auto_shard=True,
        )

    def __call__(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        return x


class TtnnSPPF:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            auto_shard=True,
        )

    def __call__(self, x):
        y = [self.cv1(x)]
        y[0] = ttnn.sharded_to_interleaved(y[0], ttnn.L1_MEMORY_CONFIG)
        y[0] = ttnn.to_layout(y[0], ttnn.ROW_MAJOR_LAYOUT)

        for i in range(3):
            tt_max = y[-1]
            tt_out = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=x.shape[0],
                input_h=20,
                input_w=20,
                channels=320,
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            y.append(tt_out)

        out = ttnn.concat(y, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv2(out)
        return out


class TtnnCIB:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.conv0 = Conv(
            device,
            parameters.cv1[0],
            self.conv_pt.cv1[0],
        )

        self.conv1 = Conv(
            device,
            parameters.cv1[1],
            self.conv_pt.cv1[1],
        )

        self.conv2 = Conv(
            device,
            parameters.cv1[2],
            self.conv_pt.cv1[2],
            auto_shard=True,
        )

        self.conv3 = Conv(
            device,
            parameters.cv1[3],
            self.conv_pt.cv1[3],
        )

        self.conv4 = Conv(
            device,
            parameters.cv1[4],
            self.conv_pt.cv1[4],
        )

    def __call__(self, x):
        input_tensor = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = input_tensor + x
        return x


class TtnnAttention:
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, device=None, parameters=None, conv_pt=None):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.qkv = Conv(
            device,
            parameters.qkv,
            self.conv_pt.qkv,
            enable_identity=True,
        )
        self.proj = Conv(
            device,
            parameters.proj,
            self.conv_pt.proj,
            enable_identity=True,
        )

        self.pe = Conv(
            device,
            parameters.pe,
            self.conv_pt.pe,
            enable_identity=True,
        )

    def __call__(self, x):
        B, C, H, W = (1, x.shape[-1], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])))
        N = H * W

        qkv = self.qkv(x)
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))
        qkv = ttnn.reshape(qkv, (1, qkv.shape[1], int(math.sqrt(qkv.shape[-1])), int(math.sqrt(qkv.shape[-1]))))
        qkv = ttnn.reshape(qkv, (B, self.num_heads, self.key_dim * 2 + self.head_dim, N))

        qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)

        q, k, v = qkv[:, :, :32, :], qkv[:, :, 32:64, :], qkv[:, :, 64:, :]

        q = ttnn.permute(q, (0, 1, 3, 2))

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)

        attn = ttnn.matmul(q, k)
        attn = attn * self.scale

        attn = ttnn.softmax(attn, dim=-1)

        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        attn = ttnn.permute(attn, (0, 1, 3, 2))
        attn = ttnn.matmul(v, attn)
        attn = ttnn.reshape(attn, (B, C, H, W))

        v = ttnn.reshape(v, (B, C, H, W))

        v = ttnn.permute(v, (0, 2, 3, 1))
        v = ttnn.reshape(v, (1, 1, v.shape[0] * v.shape[1] * v.shape[2], v.shape[3]))

        v = self.pe(v)

        attn = ttnn.permute(attn, (0, 2, 3, 1))
        attn = ttnn.reshape(attn, (1, 1, attn.shape[0] * attn.shape[1] * attn.shape[2], attn.shape[3]))

        x = attn + v

        x = self.proj(x)

        return x


class TtnnPSA:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
        )

        self.attn = TtnnAttention(
            dim=320,
            num_heads=5,
            attn_ratio=0.5,
            device=self.device,
            parameters=self.parameters.attn,
            conv_pt=self.conv_pt.attn,
        )

        self.ffn_0 = Conv(
            device,
            parameters.ffn[0],
            self.conv_pt.ffn[0],
        )

        self.ffn_1 = Conv(
            device,
            parameters.ffn[1],
            self.conv_pt.ffn[1],
            enable_identity=True,
        )

    def __call__(self, x):
        x = self.cv1(x)
        a = x[:, :, :, : x.shape[-1] // 2]
        b = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        out = self.attn(b)

        b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
        b = b + out

        out = self.ffn_0(b)
        out = self.ffn_1(out)

        b = b + out

        out = ttnn.concat([a, b], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv2(out)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

        return out


class TtnnC2f:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )
        self.cv2 = Conv(device, parameters.cv2, self.conv_pt.cv2, auto_shard=True)

        self.m = [
            TtnnBottleNeck(self.shortcut, device=self.device, parameters=self.parameters[_], conv_pt=self.conv_pt.m[_])
            for _ in range(n)
        ]

    def __call__(self, x):
        x = self.cv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x[:, :, :, : x.shape[-1] // 2]
        x2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        x1 = ttnn.from_device(x1)
        x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)
        x1 = ttnn.to_dtype(x1, dtype=ttnn.bfloat8_b)
        x1 = ttnn.to_device(x1, self.device)

        x2 = ttnn.from_device(x2)
        x2 = ttnn.to_layout(x2, ttnn.TILE_LAYOUT)
        x2 = ttnn.to_dtype(x2, dtype=ttnn.bfloat8_b)
        x2 = ttnn.to_device(x2, self.device)
        y = [x1, x2]

        for m in self.m:
            out = m(y[-1])
            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = ttnn.concat(y, -1)

        out = self.cv2(out)

        return out


class TtnnC2fCIB:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.shortcut = shortcut

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
            auto_shard=True,
        )
        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            auto_shard=True,
        )

        self.m = [
            TtnnCIB(
                shortcut=self.shortcut,
                device=self.device,
                parameters=self.parameters[2],
                conv_pt=self.conv_pt.m[_],
            )
            for _ in range(n)
        ]

    def __call__(self, x):
        x = self.cv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x[:, :, :, : x.shape[-1] // 2]
        x2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        y = [ttnn.to_layout(x1, ttnn.TILE_LAYOUT), ttnn.to_layout(x2, ttnn.TILE_LAYOUT)]

        for m in self.m:
            out = m(y[-1])

            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = ttnn.concat(y, -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.cv2(out)

        return x


class TtnnV10Detec:
    end2end = True

    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.device = device
        self.cv2_0_0 = Conv(
            device, parameters.one2one_cv2[0][0], self.conv_pt.one2one_cv2[0][0], is_detect=True, auto_shard=True
        )
        self.cv2_0_1 = Conv(
            device, parameters.one2one_cv2[0][1], self.conv_pt.one2one_cv2[0][1], is_detect=True, auto_shard=True
        )
        self.cv2_0_2 = TtYolov10_Conv2D(
            parameters.one2one_cv2[0][2], self.conv_pt.one2one_cv2[0][2], device=device, is_detect=True
        )

        self.cv2_1_0 = Conv(device, parameters.one2one_cv2[1][0], self.conv_pt.one2one_cv2[1][0], is_detect=True)
        self.cv2_1_1 = Conv(device, parameters.one2one_cv2[1][1], self.conv_pt.one2one_cv2[1][1], is_detect=True)
        self.cv2_1_2 = TtYolov10_Conv2D(
            parameters.one2one_cv2[1][2], self.conv_pt.one2one_cv2[1][2], device=device, is_detect=True
        )

        self.cv2_2_0 = Conv(device, parameters.one2one_cv2[2][0], self.conv_pt.one2one_cv2[2][0], is_detect=True)
        self.cv2_2_1 = Conv(device, parameters.one2one_cv2[2][1], self.conv_pt.one2one_cv2[2][1], is_detect=True)
        self.cv2_2_2 = TtYolov10_Conv2D(
            parameters.one2one_cv2[2][2], self.conv_pt.one2one_cv2[2][2], device=device, is_detect=True
        )

        self.cv3_0_0_0 = Conv(
            device, parameters.one2one_cv3[0][0][0], conv_pt.one2one_cv3[0][0][0], is_detect=True, auto_shard=True
        )
        self.cv3_0_0_1 = Conv(device, parameters.one2one_cv3[0][0][1], conv_pt.one2one_cv3[0][0][1], is_detect=True)
        self.cv3_0_1_0 = Conv(
            device, parameters.one2one_cv3[0][1][0], conv_pt.one2one_cv3[0][1][0], is_detect=True, auto_shard=True
        )
        self.cv3_0_1_1 = Conv(
            device, parameters.one2one_cv3[0][1][1], conv_pt.one2one_cv3[0][1][1], is_detect=True, auto_shard=True
        )
        self.cv3_0_2_0 = TtYolov10_Conv2D(
            parameters.one2one_cv3[0][2], conv_pt.one2one_cv3[0][2], device=device, is_detect=True
        )

        self.cv3_1_0_0 = Conv(
            device,
            parameters.one2one_cv3[1][0][0],
            conv_pt.one2one_cv3[1][0][0],
            is_detect=True,
            use_1d_systolic_array=False,
        )
        self.cv3_1_0_1 = Conv(
            device, parameters.one2one_cv3[1][0][1], conv_pt.one2one_cv3[1][0][1], is_detect=True, auto_shard=True
        )
        self.cv3_1_1_0 = Conv(device, parameters.one2one_cv3[1][1][0], conv_pt.one2one_cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = Conv(
            device, parameters.one2one_cv3[1][1][1], conv_pt.one2one_cv3[1][1][1], is_detect=True, auto_shard=True
        )
        self.cv3_1_2_0 = TtYolov10_Conv2D(
            parameters.one2one_cv3[1][2], conv_pt.one2one_cv3[1][2], device=device, is_detect=True
        )

        self.cv3_1_0_0 = Conv(
            device,
            parameters.one2one_cv3[1][0][0],
            conv_pt.one2one_cv3[1][0][0],
            is_detect=True,
            use_1d_systolic_array=True,
            auto_shard=True,
        )

        self.cv3_1_0_1 = Conv(device, parameters.one2one_cv3[1][0][1], conv_pt.one2one_cv3[1][0][1], is_detect=True)
        self.cv3_1_1_0 = Conv(device, parameters.one2one_cv3[1][1][0], conv_pt.one2one_cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = Conv(device, parameters.one2one_cv3[1][1][1], conv_pt.one2one_cv3[1][1][1], is_detect=True)
        self.cv3_1_2_0 = TtYolov10_Conv2D(
            parameters.one2one_cv3[1][2], conv_pt.one2one_cv3[1][2], device=device, is_detect=True
        )

        self.cv3_2_0_0 = Conv(
            device, parameters.one2one_cv3[2][0][0], conv_pt.one2one_cv3[2][0][0], is_detect=True, auto_shard=True
        )

        self.cv3_2_0_1 = Conv(device, parameters.one2one_cv3[2][0][1], conv_pt.one2one_cv3[2][0][1], is_detect=True)
        self.cv3_2_1_0 = Conv(device, parameters.one2one_cv3[2][1][0], conv_pt.one2one_cv3[2][1][0], is_detect=True)
        self.cv3_2_1_1 = Conv(device, parameters.one2one_cv3[2][1][1], conv_pt.one2one_cv3[2][1][1], is_detect=True)
        self.cv3_2_2_0 = TtYolov10_Conv2D(
            parameters.one2one_cv3[2][2], conv_pt.one2one_cv3[2][2], device=device, is_detect=True
        )
        self.dfl = Conv(device, parameters.dfl, self.conv_pt.dfl, is_dfl=True)

        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, y1, y2, y3):
        x1 = self.cv2_0_0(y1)
        x1 = self.cv2_0_1(x1)
        x1 = self.cv2_0_2(x1)
        x2 = self.cv2_1_0(y2)
        x2 = self.cv2_1_1(x2)
        x2 = self.cv2_1_2(x2)

        x3 = self.cv2_2_0(y3)
        x3 = self.cv2_2_1(x3)
        x3 = self.cv2_2_2(x3)

        x4 = self.cv3_0_0_0(y1)
        x4 = self.cv3_0_0_1(x4)
        x4 = self.cv3_0_1_0(x4)
        x4 = self.cv3_0_1_1(x4)
        x4 = self.cv3_0_2_0(x4)

        x5 = self.cv3_1_0_0(y2)

        x5 = self.cv3_1_0_1(x5)
        x5 = self.cv3_1_1_0(x5)
        x5 = self.cv3_1_1_1(x5)
        x5 = self.cv3_1_2_0(x5)

        x6 = self.cv3_2_0_0(y3)
        x6 = self.cv3_2_0_1(x6)
        x6 = self.cv3_2_1_0(x6)
        x6 = self.cv3_2_1_1(x6)
        x6 = self.cv3_2_2_0(x6)

        x1 = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x4 = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        x5 = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1 = ttnn.concat((x1, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.concat((x2, x5), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3 = ttnn.concat((x3, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = ttnn.concat((y1, y2, y3), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.squeeze(y, dim=0)

        ya, yb = y[:, :, :64], y[:, :, 64:144]
        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)
        ttnn.deallocate(x4)
        ttnn.deallocate(x5)
        ttnn.deallocate(x6)
        ttnn.deallocate(y)
        ya = ttnn.reshape(ya, (ya.shape[0], y.shape[1], 4, 16))
        ya = ttnn.permute(ya, (0, 2, 1, 3))
        ya = ttnn.softmax(ya, dim=-1)
        c = self.dfl(ya)
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
        c1 = anchor - c1
        c2 = anchor + c2

        z = ttnn.concat((c1, c2), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        z = ttnn.multiply(z, strides)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid_accurate(yb)
        ttnn.deallocate(c)
        ttnn.deallocate(c1)
        ttnn.deallocate(c2)
        ttnn.deallocate(anchor)
        ttnn.deallocate(strides)
        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(yb)
        ttnn.deallocate(z)
        return out


class TtnnYolov10:
    def __init__(self, device, parameters, conv_pt):
        self.device = device
        self.conv1 = Conv(device, parameters.conv_args[0], conv_pt.model[0], config_override={"act_block_h": 64})
        self.conv2 = Conv(device, parameters.conv_args[1], conv_pt.model[1], auto_shard=True)
        self.c2f_1 = TtnnC2f(
            shortcut=True, n=3, device=self.device, parameters=parameters.conv_args[2], conv_pt=conv_pt.model[2]
        )
        self.conv3 = Conv(device, parameters.conv_args[3], conv_pt.model[3], auto_shard=True)
        self.c2f_2 = TtnnC2f(
            shortcut=True, n=6, device=self.device, parameters=parameters.conv_args[4], conv_pt=conv_pt.model[4]
        )
        self.scdown_1 = TtnnSCDown(device=device, parameters=parameters.conv_args[5], conv_pt=conv_pt.model[5])
        self.c2fcib_1 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[6], n=6, conv_pt=conv_pt.model[6])
        self.scdown_2 = TtnnSCDown(device=device, parameters=parameters.conv_args[7], conv_pt=conv_pt.model[7])
        self.c2fcib_2 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[8], conv_pt=conv_pt.model[8])
        self.sppf = TtnnSPPF(device=device, parameters=parameters.conv_args[9], conv_pt=conv_pt.model[9])
        self.psa_1 = TtnnPSA(device=device, parameters=parameters.conv_args[10], conv_pt=conv_pt.model[10])
        self.c2fcib_3 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[13], conv_pt=conv_pt.model[13])
        self.c2f_3 = TtnnC2f(
            shortcut=False, n=3, device=self.device, parameters=parameters.conv_args[16], conv_pt=conv_pt.model[16]
        )
        self.conv4 = Conv(device, parameters.conv_args[17], conv_pt.model[17], auto_shard=True)
        self.c2fcib_4 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[19], conv_pt=conv_pt.model[19])
        self.scdown_3 = TtnnSCDown(device=device, parameters=parameters.conv_args[20], conv_pt=conv_pt.model[20])
        self.c2fcib_5 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[22], conv_pt=conv_pt.model[22])
        self.detect = TtnnV10Detec(device=device, parameters=parameters.model_args.model[23], conv_pt=conv_pt.model[23])

    def __call__(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.c2f_1(x1)
        x3 = self.conv3(x2)
        x4 = self.c2f_2(x3)
        x5 = self.scdown_1(x4)
        x6 = self.c2fcib_1(x5)
        x7 = self.scdown_2(x6)
        x8 = self.c2fcib_2(x7)
        x9 = self.sppf(x8)
        x10 = self.psa_1(x9)
        ttnn.deallocate(x)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)
        ttnn.deallocate(x5)
        ttnn.deallocate(x7)
        ttnn.deallocate(x8)
        ttnn.deallocate(x9)

        x10 = interleaved_to_sharded(x10)
        x11 = ttnn.upsample(x10, scale_factor=2)

        if x11.is_sharded():
            x11 = ttnn.sharded_to_interleaved(x11, memory_config=ttnn.L1_MEMORY_CONFIG)
        x11 = ttnn.reshape(x11, (1, 1, x11.shape[0] * x11.shape[1] * x11.shape[2], x11.shape[3]))
        x11 = ttnn.to_layout(x11, layout=ttnn.ROW_MAJOR_LAYOUT)
        x12 = ttnn.concat((x11, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x6)
        ttnn.deallocate(x11)
        x13 = self.c2fcib_3(x12)
        ttnn.deallocate(x12)

        x13 = interleaved_to_sharded(x13)
        x14 = ttnn.upsample(x13, scale_factor=2)

        if x14.is_sharded():
            x14 = ttnn.sharded_to_interleaved(x14, memory_config=ttnn.L1_MEMORY_CONFIG)
        x14 = ttnn.reshape(x14, (1, 1, x14.shape[0] * x14.shape[1] * x14.shape[2], x14.shape[3]))
        x14 = ttnn.to_layout(x14, layout=ttnn.ROW_MAJOR_LAYOUT)
        x15 = ttnn.concat((x14, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x14)
        ttnn.deallocate(x4)

        x16 = self.c2f_3(x15)
        ttnn.deallocate(x15)
        x17 = self.conv4(x16)

        if x17.is_sharded():
            x17 = ttnn.sharded_to_interleaved(x17, memory_config=ttnn.L1_MEMORY_CONFIG)
        x17 = ttnn.reshape(x17, (1, 1, x17.shape[0] * x17.shape[1] * x17.shape[2], x17.shape[3]))
        x17 = ttnn.to_layout(x17, layout=ttnn.ROW_MAJOR_LAYOUT)
        x13 = ttnn.reshape(x13, ((1, 1, x13.shape[0] * x13.shape[1] * x13.shape[2], x13.shape[3])))
        x13 = ttnn.sharded_to_interleaved(x13, memory_config=ttnn.L1_MEMORY_CONFIG)
        x18 = ttnn.concat((x17, x13), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x4)
        ttnn.deallocate(x17)
        ttnn.deallocate(x13)

        x19 = self.c2fcib_4(x18)
        x20 = self.scdown_3(x19)
        ttnn.deallocate(x18)

        if x20.is_sharded():
            x20 = ttnn.sharded_to_interleaved(x20, memory_config=ttnn.L1_MEMORY_CONFIG)
        x20 = ttnn.reshape(x20, (1, 1, x20.shape[0] * x20.shape[1] * x20.shape[2], x20.shape[3]))
        x20 = ttnn.to_layout(x20, layout=ttnn.ROW_MAJOR_LAYOUT)
        x10 = ttnn.reshape(x10, ((1, 1, x10.shape[0] * x10.shape[1] * x10.shape[2], x10.shape[3])))
        x10 = ttnn.sharded_to_interleaved(x10, memory_config=ttnn.L1_MEMORY_CONFIG)

        x21 = ttnn.concat((x20, x10), -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        x22 = self.c2fcib_5(x21)
        x23 = self.detect(x16, x19, x22)
        return x23
