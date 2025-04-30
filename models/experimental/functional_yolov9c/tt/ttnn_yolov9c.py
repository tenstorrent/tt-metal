# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.experimental.yolo_common.yolo_utils import concat, determine_num_cores, get_core_grid_from_num_cores


def interleaved_to_sharded(x):
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    nhw = x.shape[0] * x.shape[1] * x.shape[2]
    num_cores = determine_num_cores(nhw, int(math.sqrt(x.shape[2])))
    num_cores = 56 if num_cores == 1 else num_cores
    core_grid = get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    return ttnn.reshard(x, shardspec) if x.is_sharded() else ttnn.interleaved_to_sharded(x, shardspec)


class TtYOLOv9cConv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_detect=False,
        is_dfl=False,
        config_override=None,
        deallocate_activation=False,
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
        self.deallocate_activation = deallocate_activation
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
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
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        hw = output_height * output_width
        if x.shape[2] != hw:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = x[:, :, :hw, :]
        return x


class TtnnRepconv:
    def __init__(self, device, parameter, conv_pt):
        self.conv1 = TtYOLOv9cConv2D(
            device=device, conv=parameter.conv1.conv, conv_pth=conv_pt.conv1.conv, activation=""
        )
        self.conv2 = TtYOLOv9cConv2D(
            device=device, conv=parameter.conv2.conv, conv_pth=conv_pt.conv2.conv, activation=""
        )

    def __call__(self, device, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        x = ttnn.silu(conv1_out + conv2_out)

        return x


class TtnnRepBottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtnnRepconv(device=device, parameter=parameter.cv1, conv_pt=conv_pt.cv1)
        self.cv2 = TtYOLOv9cConv2D(device=device, conv=parameter.cv2.conv, conv_pth=conv_pt.cv2.conv, activation="silu")

    def __call__(self, device, x):
        input = x
        x = self.cv1(device, x)
        x = self.cv2(x)
        return input + x


class TtnnRepcsp:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtYOLOv9cConv2D(device=device, conv=parameter.cv1.conv, conv_pth=conv_pt.cv1.conv, activation="silu")
        self.cv2 = TtYOLOv9cConv2D(device=device, conv=parameter.cv2.conv, conv_pth=conv_pt.cv2.conv, activation="silu")
        self.cv3 = TtYOLOv9cConv2D(device=device, conv=parameter.cv3.conv, conv_pth=conv_pt.cv3.conv, activation="silu")
        self.m = TtnnRepBottleneck(device, parameter.m[0], conv_pt.m[0])

    def __call__(self, device, x):
        cv1_out = self.cv1(x)
        m_out = self.m(device, cv1_out)
        ttnn.deallocate(cv1_out)

        cv2_out = self.cv2(x)

        cv2_out = ttnn.to_layout(cv2_out, ttnn.ROW_MAJOR_LAYOUT)
        cv2_out = ttnn.to_dtype(cv2_out, ttnn.bfloat16)

        m_out = ttnn.to_layout(m_out, ttnn.ROW_MAJOR_LAYOUT)
        m_out = ttnn.to_dtype(m_out, ttnn.bfloat16)

        concat_out = concat(-1, True, m_out, cv2_out)
        ttnn.deallocate(m_out)
        ttnn.deallocate(cv2_out)

        x = self.cv3(concat_out)
        ttnn.deallocate(concat_out)

        return x


class TtnnRepncspelan4:
    def __init__(self, device, parameter, conv_pt, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED):
        self.cv1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv1.conv,
            conv_pth=conv_pt.cv1.conv,
            activation="silu",
            shard_layout=shard_layout,
        )
        self.k1 = TtnnRepcsp(device, parameter.cv2[0], conv_pt.cv2[0])
        self.k2 = TtYOLOv9cConv2D(
            device=device, conv=parameter.cv2[1].conv, conv_pth=conv_pt.cv2[1].conv, activation="silu"
        )
        self.k3 = TtnnRepcsp(device, parameter.cv3[0], conv_pt.cv3[0])
        self.k4 = TtYOLOv9cConv2D(
            device=device, conv=parameter.cv3[1].conv, conv_pth=conv_pt.cv3[1].conv, activation="silu"
        )
        self.cv4 = TtYOLOv9cConv2D(device=device, conv=parameter.cv4.conv, conv_pth=conv_pt.cv4.conv, activation="silu")

    def __call__(self, device, x):
        x = self.cv1(x)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1 = x[:, :, :, : x.shape[-1] // 2]
        y2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]

        y2 = interleaved_to_sharded(y2)

        cv2_out = self.k1(device, y2)
        cv3_out = self.k2(cv2_out)
        ttnn.deallocate(cv2_out)

        cv4_out = self.k3(device, cv3_out)
        cv5_out = self.k4(cv4_out)
        ttnn.deallocate(cv4_out)

        x = concat(-1, True, y1, y2, cv3_out, cv5_out)

        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(cv3_out)
        ttnn.deallocate(cv5_out)
        x = self.cv4(x)

        return x


class TtnnADown:
    def __init__(self, device, parameter, conv_pt, use_1d_systolic_array=True):
        self.parameter = parameter
        self.cv1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv1.conv,
            conv_pth=conv_pt.cv1.conv,
            activation="silu",
            use_1d_systolic_array=use_1d_systolic_array,
        )
        self.cv2 = TtYOLOv9cConv2D(device=device, conv=parameter.cv2.conv, conv_pth=conv_pt.cv2.conv, activation="silu")

    def __call__(self, device, x):
        x = ttnn.avg_pool2d(
            input_tensor=x,
            batch_size=x.shape[0],
            input_h=int(math.sqrt(x.shape[-2])),
            input_w=int(math.sqrt(x.shape[-2])),
            channels=x.shape[-1],
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        x1 = x[:, :, :, : x.shape[-1] // 2]
        x2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        ttnn.deallocate(x)
        x1 = self.cv1(x1)

        x2 = ttnn.reshape(x2, (1, 1, x2.shape[0] * x2.shape[1] * x2.shape[2], x2.shape[-1]))
        x2_maxpool = ttnn.max_pool2d(
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
        x2 = self.cv2(x2_maxpool)
        x = concat(-1, True, x1, x2)

        ttnn.deallocate(x1)
        ttnn.deallocate(x2)

        return x


class TtnnSPPELAN:
    def __init__(self, device, parameter, conv_pt):
        self.parameter = parameter
        self.cv1 = TtYOLOv9cConv2D(device=device, conv=parameter.cv1.conv, conv_pth=conv_pt.cv1.conv, activation="silu")
        self.cv5 = TtYOLOv9cConv2D(device=device, conv=parameter.cv5.conv, conv_pth=conv_pt.cv5.conv, activation="silu")

    def __call__(self, device, x):
        x = self.cv1(x)
        x1 = x
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x)

        TILE_WIDTH = 32
        in_c = self.parameter.cv5.conv.in_channels
        in_c_padded = in_c
        if in_c % TILE_WIDTH != 0 and in_c != 16:
            in_c_padded = in_c + (TILE_WIDTH - in_c % TILE_WIDTH)
        m1 = ttnn.max_pool2d(
            x,
            batch_size=self.parameter.cv5.conv.batch_size,
            input_h=self.parameter.cv5.conv.input_height,
            input_w=self.parameter.cv5.conv.input_width,
            channels=in_c_padded,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        m2 = ttnn.max_pool2d(
            m1,
            batch_size=self.parameter.cv5.conv.batch_size,
            input_h=self.parameter.cv5.conv.input_height,
            input_w=self.parameter.cv5.conv.input_width,
            channels=in_c_padded,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m3 = ttnn.max_pool2d(
            m2,
            batch_size=self.parameter.cv5.conv.batch_size,
            input_h=self.parameter.cv5.conv.input_height,
            input_w=self.parameter.cv5.conv.input_width,
            channels=in_c_padded,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )

        y = concat(-1, True, x1, m1, m2, m3)
        ttnn.deallocate(x1)
        ttnn.deallocate(m1)
        ttnn.deallocate(m2)
        ttnn.deallocate(m3)

        x = self.cv5(y)

        return x


class TtnnDetect:
    def __init__(self, device, parameter, conv_pt):
        self.cv2_0_0 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv2[0][0].conv,
            conv_pth=conv_pt.cv2[0][0].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv2_0_1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv2[0][1].conv,
            conv_pth=conv_pt.cv2[0][1].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv2_0_2 = TtYOLOv9cConv2D(
            conv=parameter.cv2[0][2], conv_pth=conv_pt.cv2[0][2], device=device, is_detect=True
        )

        self.cv2_1_0 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv2[1][0].conv,
            conv_pth=conv_pt.cv2[1][0].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv2_1_1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv2[1][1].conv,
            conv_pth=conv_pt.cv2[1][1].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv2_1_2 = TtYOLOv9cConv2D(
            conv=parameter.cv2[1][2], conv_pth=conv_pt.cv2[1][2], device=device, is_detect=True
        )

        self.cv2_2_0 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv2[2][0].conv,
            conv_pth=conv_pt.cv2[2][0].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv2_2_1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv2[2][1].conv,
            conv_pth=conv_pt.cv2[2][1].conv,
            is_detect=True,
        )
        self.cv2_2_2 = TtYOLOv9cConv2D(
            conv=parameter.cv2[2][2], conv_pth=conv_pt.cv2[2][2], device=device, is_detect=True
        )

        self.cv3_0_0 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv3[0][0].conv,
            conv_pth=conv_pt.cv3[0][0].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv3_0_1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv3[0][1].conv,
            conv_pth=conv_pt.cv3[0][1].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv3_0_2 = TtYOLOv9cConv2D(
            conv=parameter.cv3[0][2], conv_pth=conv_pt.cv3[0][2], device=device, is_detect=True
        )

        self.cv3_1_0 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv3[1][0].conv,
            conv_pth=conv_pt.cv3[1][0].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv3_1_1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv3[1][1].conv,
            conv_pth=conv_pt.cv3[1][1].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv3_1_2 = TtYOLOv9cConv2D(
            conv=parameter.cv3[1][2], conv_pth=conv_pt.cv3[1][2], device=device, is_detect=True
        )

        self.cv3_2_0 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv3[2][0].conv,
            conv_pth=conv_pt.cv3[2][0].conv,
            activation="silu",
            is_detect=True,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.cv3_2_1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameter.cv3[2][1].conv,
            conv_pth=conv_pt.cv3[2][1].conv,
            activation="silu",
            is_detect=True,
        )
        self.cv3_2_2 = TtYOLOv9cConv2D(
            conv=parameter.cv3[2][2], conv_pth=conv_pt.cv3[2][2], device=device, is_detect=True
        )
        self.dfl = TtYOLOv9cConv2D(device=device, conv=parameter.dfl.conv, conv_pth=conv_pt.dfl.conv, is_dfl=True)
        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, device, y1, y2, y3):
        x1 = self.cv2_0_0(y1)
        x1 = self.cv2_0_1(x1)
        x1 = self.cv2_0_2(x1)
        x2 = self.cv2_1_0(y2)
        x2 = self.cv2_1_1(x2)
        x2 = self.cv2_1_2(x2)
        x3 = self.cv2_2_0(y3)
        x3 = self.cv2_2_1(x3)
        x3 = self.cv2_2_2(x3)
        x4 = self.cv3_0_0(y1)
        x4 = self.cv3_0_1(x4)
        x4 = self.cv3_0_2(x4)
        x5 = self.cv3_1_0(y2)
        x5 = self.cv3_1_1(x5)
        x5 = self.cv3_1_2(x5)
        x6 = self.cv3_2_0(y3)
        x6 = self.cv3_2_1(x6)
        x6 = self.cv3_2_2(x6)

        y1 = concat(-1, False, x1, x4)
        y2 = concat(-1, False, x2, x5)
        y3 = concat(-1, False, x3, x6)

        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)
        ttnn.deallocate(x4)
        ttnn.deallocate(x5)
        ttnn.deallocate(x6)

        y = concat(2, False, y1, y2, y3)

        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)

        ya, yb = y[:, :, :, :64], y[:, :, :, 64:144]
        ya = ttnn.permute(ya, (0, 1, 3, 2))
        ya = ttnn.reshape(ya, (ya.shape[0], 4, 16, ya.shape[-1]))
        ya = ttnn.permute(ya, (0, 2, 1, 3))
        ya = ttnn.to_layout(ya, ttnn.TILE_LAYOUT)
        ya = ttnn.softmax(ya, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ya = ttnn.permute(ya, (0, 2, 3, 1))
        c = self.dfl(ya)

        if c.is_sharded():
            c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]

        c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)

        c1 = self.anchors - c1
        c2 = self.anchors + c2
        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)

        ttnn.deallocate(ya)
        ttnn.deallocate(c)
        ttnn.deallocate(c1)
        ttnn.deallocate(c2)

        z = concat(1, False, z2, z1)
        z = ttnn.multiply(z, self.strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        yb = ttnn.squeeze(yb, 0)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)

        out = concat(1, False, z, yb)

        ttnn.deallocate(z)
        ttnn.deallocate(yb)

        return out


class YoloV9:
    def __init__(self, device, parameters):
        self.device = device
        self.conv1 = TtYOLOv9cConv2D(
            device=device,
            conv=parameters.conv_args[0].conv,
            conv_pth=parameters.model[0].conv,
            config_override={"act_block_h": 32},
            activation="silu",
            deallocate_activation=True,
        )  # 0
        self.conv2 = TtYOLOv9cConv2D(
            device=device, conv=parameters.conv_args[1].conv, conv_pth=parameters.model[1].conv, activation="silu"
        )  # 1
        self.repncspelan4_1 = TtnnRepncspelan4(device, parameters.conv_args[2], parameters.model[2])  # 2
        self.adown_1 = TtnnADown(device, parameters.conv_args[3], parameters.model[3], use_1d_systolic_array=False)  # 3
        self.repncspelan4_2 = TtnnRepncspelan4(device, parameters.conv_args[4], parameters.model[4])  # 4
        self.adown_2 = TtnnADown(device, parameters.conv_args[5], parameters.model[5])  # 5
        self.repncspelan4_3 = TtnnRepncspelan4(device, parameters.conv_args[6], parameters.model[6])  # 6
        self.adown_3 = TtnnADown(device, parameters.conv_args[7], parameters.model[7])  # 7
        self.repncspelan4_4 = TtnnRepncspelan4(device, parameters.conv_args[8], parameters.model[8])  # 8
        self.ttnn_sppelan = TtnnSPPELAN(device, parameters.conv_args[9], parameters.model[9])  # 9
        self.repncspelan4_5 = TtnnRepncspelan4(device, parameters.conv_args[12], parameters.model[12])  # 12
        self.repncspelan4_6 = TtnnRepncspelan4(device, parameters.conv_args[15], parameters.model[15])  # 15
        self.adown_6 = TtnnADown(device, parameters.conv_args[16], parameters.model[16])  # 16
        self.repncspelan4_7 = TtnnRepncspelan4(device, parameters.conv_args[18], parameters.model[18])  # 18
        self.adown_7 = TtnnADown(device, parameters.conv_args[19], parameters.model[19])  # 19
        self.repncspelan4_8 = TtnnRepncspelan4(device, parameters.conv_args[21], parameters.model[21])  # 21
        self.detect = TtnnDetect(device, parameters.model_args.model[22], parameters.model[22])  # 22

    def __call__(self, x):
        x = self.conv1(x)  # 0
        x = self.conv2(x)  # 1
        x = self.repncspelan4_1(self.device, x)  # 2
        x = self.adown_1(self.device, x)  # 3
        x = self.repncspelan4_2(self.device, x)  # 4
        x4 = x
        x = self.adown_2(self.device, x)  # 5
        x = self.repncspelan4_3(self.device, x)  # 6
        x6 = x
        x = self.adown_3(self.device, x)  # 7
        x = self.repncspelan4_4(self.device, x)  # 8
        x = self.ttnn_sppelan(self.device, x)  # 9
        x10 = x
        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)  # 10

        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        x = concat(-1, True, x, x6)  # 11
        ttnn.deallocate(x6)

        x = self.repncspelan4_5(self.device, x)  # 12
        x13 = x
        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)  # 13

        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        x = concat(-1, True, x, x4)  # 14
        x = ttnn.sharded_to_interleaved(x)
        ttnn.deallocate(x4)

        x = self.repncspelan4_6(self.device, x)  # 15
        x16 = x
        x = self.adown_6(self.device, x)  # 16

        x = concat(-1, False, x, x13)  # 17
        ttnn.deallocate(x13)

        x = self.repncspelan4_7(self.device, x)  # 18
        x19 = x
        x = self.adown_7(self.device, x)  # 19
        x = concat(-1, True, x, x10)  # 20
        ttnn.deallocate(x10)

        x = self.repncspelan4_8(self.device, x)  # 21
        x22 = x
        x = self.detect(self.device, x16, x19, x22)  # 22

        ttnn.deallocate(x16)
        ttnn.deallocate(x19)
        ttnn.deallocate(x22)

        return x
