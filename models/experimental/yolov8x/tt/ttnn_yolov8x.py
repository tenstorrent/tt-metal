# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

from models.experimental.yolov8x.tt.ttnn_yolov8x_utils import (
    ttnn_decode_bboxes,
)
from models.experimental.yolo_common.yolo_utils import (
    determine_num_cores,
    get_core_grid_from_num_cores,
)


def sharded_concat(input_tensors, num_cores=64, dim=3):  # expected input tensors to be in fp16, RM, same (h*w)
    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_configs = []

    for i in range(len(input_tensors)):
        input_sharded_memory_config = ttnn.create_sharded_memory_config(
            (shard_height, input_tensors[i].shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_configs.append(input_sharded_memory_config)

    sharded_inputs = [
        ttnn.to_memory_config(tensor, config) for tensor, config in zip(input_tensors, input_sharded_memory_configs)
    ]

    total_width = sum(tensor.shape[-1] for tensor in input_tensors)
    out_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, total_width),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    output = ttnn.concat(sharded_inputs, dim, memory_config=out_sharded_memory_config)
    output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


class TtConv:
    def __init__(
        self,
        device,
        parameters,
        path,
        input_params,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=None,
        bfloat8=True,
        change_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        is_fused=True,
        is_detect_cv2=False,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=False,
        batch_size=1,
    ):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.bfloat8 = bfloat8
        self.change_shard = change_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.is_fused = is_fused
        self.is_detect_cv2 = is_detect_cv2
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.batch_size = batch_size

        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters[path]

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation="",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=False,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
        )

        if self.deallocate_activation:
            conv_config.deallocate_activation = self.deallocate_activation

        if self.change_shard:
            conv_config.shard_layout = None

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        if self.width_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        if self.bfloat8:
            conv_config.weights_dtype = ttnn.bfloat8_b

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
            input_height = int(math.sqrt(x.shape[2]) // self.batch_size)
            input_width = int(math.sqrt(x.shape[2]) // self.batch_size)

        [x, [out_height, out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=self.input_params[4],
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
            groups=self.groups,
            memory_config=None,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        if self.is_detect_cv2:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            return x, out_height, out_width

        x = ttnn.silu(x)
        return x, out_height, out_width


class TtBottleneck:
    def __init__(
        self,
        device,
        parameters,
        path,
        shortcut,
        change_shard,
        input_params,
        act_block_h=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        tilize=False,
    ):
        self.device = device
        self.path = path
        self.tilize = tilize
        self.shortcut = shortcut
        self.block_shard = block_shard
        self.cv1 = TtConv(
            device,
            parameters,
            f"{self.path}.cv1",
            input_params,
            change_shard=change_shard,
            block_shard=self.block_shard,
            deallocate_activation=deallocate_activation,
            output_layout=output_layout,
        )
        self.cv2 = TtConv(
            device,
            parameters,
            f"{self.path}.cv2",
            input_params,
            act_block_h=act_block_h,
            change_shard=change_shard,
            block_shard=self.block_shard,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        cv2, out_h, out_w = self.cv2(cv1)  # pass cv1
        ttnn.deallocate(cv1)

        if self.tilize:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv2


class TtC2f:
    def __init__(
        self,
        device,
        parameters,
        path,
        n=1,
        shortcut=False,
        change_shard=None,
        input_params=None,
        act_block_h=False,
        bfloat8=True,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    ):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.n = n
        self.shortcut = shortcut
        self.change_shard = change_shard
        self.input_params = input_params
        self.act_block_h = act_block_h
        self.bfloat8 = bfloat8
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout

        self.cv1 = TtConv(
            device,
            self.parameters,
            f"{self.path}.cv1",
            input_params=self.input_params[0],
            bfloat8=self.bfloat8,
            change_shard=self.change_shard,
            deallocate_activation=self.deallocate_activation,
            output_layout=self.output_layout,
        )

        self.cv2 = TtConv(
            self.device,
            self.parameters,
            f"{self.path}.cv2",
            input_params=self.input_params[1],
            bfloat8=self.bfloat8,
            block_shard=self.block_shard,
            change_shard=self.change_shard,
            deallocate_activation=self.deallocate_activation,
        )

        self.bottleneck_modules = []
        for i in range(self.n):
            self.tilize = i == 0
            self.bottleneck_modules.append(
                TtBottleneck(
                    self.device,
                    self.parameters,
                    f"{self.path}.m.{i}",
                    self.shortcut,
                    self.change_shard,
                    input_params=self.input_params[2],
                    act_block_h=self.act_block_h,
                    block_shard=self.block_shard,
                    deallocate_activation=self.deallocate_activation,
                    tilize=self.tilize,
                )
            )

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = [
            cv1[:, :, :, : cv1.shape[-1] // 2],
            cv1[:, :, :, cv1.shape[-1] // 2 :],
        ]

        ttnn.deallocate(cv1)

        for i in range(self.n):
            z = self.bottleneck_modules[i](y[-1])
            y.append(z)

        y[0] = ttnn.to_layout(y[0], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        y[1] = ttnn.to_layout(y[1], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        if not self.shortcut:
            for i in range(2, len(y)):
                y[i] = ttnn.sharded_to_interleaved(y[i], ttnn.L1_MEMORY_CONFIG)

        x = ttnn.concat(y, 3)

        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x, out_h, out_w = self.cv2(x)
        return x, out_h, out_w


class TtSppf:
    def __init__(self, device, parameters, path, input_params, batch_size):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.batch_size = batch_size

        self.cv1 = TtConv(device, parameters, f"{path}.cv1", input_params=input_params[0], change_shard=True)
        self.cv2 = TtConv(device, parameters, f"{path}.cv2", input_params=input_params[1], change_shard=True)

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
        y = [cv1]

        for i in range(3):
            output = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=self.batch_size,
                input_h=out_h,
                input_w=out_w,
                channels=y[-1].shape[-1],
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
            )
            y.append(output)

        x = sharded_concat(y)
        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x, out_h, out_w = self.cv2(x)
        return x, out_h, out_w


class TtDetectCv2:
    def __init__(self, device, parameters, path, input_params):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.conv0 = TtConv(device, parameters, f"{path}.0", input_params=input_params[0], bfloat8=True)
        self.conv1 = TtConv(device, parameters, f"{path}.1", input_params=input_params[1], bfloat8=True)
        self.conv2 = TtConv(
            device,
            parameters,
            path,
            input_params=input_params[2],
            bfloat8=True,
            is_fused=False,
            change_shard=True,
            is_detect_cv2=True,
        )

    def __call__(self, x):
        x, out_h, out_w = self.conv0(x)
        x, out_h, out_w = self.conv1(x)
        x, out_h, out_w = self.conv2(x)
        return x, out_h, out_w


class TtDFL:
    def __init__(self, device, parameters, path, input_params):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.conv = TtConv(device, parameters, path, input_params, bfloat8=True, is_fused=False, change_shard=True)

    def __call__(self, x, c1=16):
        b, _, a = x.shape

        x = ttnn.reshape(x, (b, 4, c1, a), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.softmax(x, dim=2)
        x = ttnn.permute(x, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        x, h, w = self.conv(x)
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, (x.shape[0], 1, 4, int(x.shape[3] / 4)))
        x = ttnn.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        return x


class TtDetect:
    def __init__(self, device, parameters, path, input_params, nc=80, ch=(320, 640, 640)):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.nc = nc
        self.ch = ch
        self.detect_cv2_modules = []
        self.detect_cv3_modules = []

        nl = len(self.ch)
        for i in range(nl):
            cv2_params = input_params["cv2_params"][i]["input_params"]
            cv3_params = input_params["cv3_params"][i]["input_params"]
            self.detect_cv2_modules.append(TtDetectCv2(device, parameters, f"{path}.cv2.{i}", input_params=cv2_params))
            self.detect_cv3_modules.append(TtDetectCv2(device, parameters, f"{path}.cv3.{i}", input_params=cv3_params))

        self.dfl_module = TtDFL(
            device, parameters, f"{path}.dfl", input_params=input_params["dfl_params"]["input_params"]
        )

    def __call__(self, x, nc=80, ch=(), reg_max=16):
        nc = self.nc
        ch = self.ch
        nl = len(ch)
        no = nc + reg_max * 4

        for i in range(nl):
            a, _, _ = self.detect_cv2_modules[i](x[i])
            b, _, _ = self.detect_cv3_modules[i](x[i])
            x[i] = ttnn.concat((a, b), dim=3)

        shape = x[0].shape
        anchors, strides = self.parameters["anchors"], self.parameters["strides"]

        xi = []
        for i in x:
            i = ttnn.reshape(i, (shape[0], -1, no), memory_config=ttnn.L1_MEMORY_CONFIG)
            xi.append(i)

        x_cat = ttnn.concat(xi, 1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_cat = ttnn.permute(x_cat, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

        box = ttnn.slice(x_cat, [0, 0, 0], [1, 64, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)
        cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)
        dfl = self.dfl_module(box)
        dbox = ttnn_decode_bboxes(self.device, dfl, anchors)
        dbox = dbox * strides

        return [ttnn.concat((dbox, ttnn.sigmoid(cls)), dim=1), x]


class TtDetectionModel:
    def __init__(self, device, parameters, res=(640, 640), batch_size=1, reg_max=16):
        self.device = device
        self.parameters = parameters
        self.res = res
        self.reg_max = reg_max
        self.batch_size = batch_size
        sppf_configs = {"input_params": ((1, 1, 0, 320, 640), (1, 1, 0, 640, 1280))}
        c2f_configs = {
            "model.2": {"input_params": ((1, 1, 0, 160, 160), (1, 1, 0, 160, 400), (3, 1, 1, 80, 80))},
            "model.4": {"input_params": ((1, 1, 0, 320, 320), (1, 1, 0, 320, 1280), (3, 1, 1, 160, 160))},
            "model.6": {"input_params": ((1, 1, 0, 640, 640), (1, 1, 0, 640, 2560), (3, 1, 1, 320, 320))},
            "model.8": {"input_params": ((1, 1, 0, 640, 640), (1, 1, 0, 640, 1600), (3, 1, 1, 320, 320))},
            "model.12": {"input_params": ((1, 1, 0, 640, 1280), (1, 1, 0, 640, 1600), (3, 1, 1, 320, 320))},
            "model.15": {"input_params": ((1, 1, 0, 320, 960), (1, 1, 0, 320, 800), (3, 1, 1, 160, 160))},
            "model.18": {"input_params": ((1, 1, 0, 640, 960), (1, 1, 0, 640, 1600), (3, 1, 1, 320, 320))},
            "model.21": {"input_params": ((1, 1, 0, 640, 1280), (1, 1, 0, 640, 1600), (3, 1, 1, 320, 320))},
        }
        detect_config = {
            "cv2_params": [
                {
                    "input_params": [
                        (3, 1, 1, 80, 320),
                        (3, 1, 1, 80, 80),
                        (1, 1, 0, 64, 80),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 80, 640),
                        (3, 1, 1, 80, 80),
                        (1, 1, 0, 64, 80),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 80, 640),
                        (3, 1, 1, 80, 80),
                        (1, 1, 0, 64, 80),
                    ]
                },
            ],
            "cv3_params": [
                {
                    "input_params": [
                        (3, 1, 1, 320, 320),
                        (3, 1, 1, 320, 320),
                        (1, 1, 0, 80, 320),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 320, 640),
                        (3, 1, 1, 320, 320),
                        (1, 1, 0, 80, 320),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 320, 640),
                        (3, 1, 1, 320, 320),
                        (1, 1, 0, 80, 320),
                    ]
                },
            ],
            "dfl_params": {
                "input_params": (1, 1, 0, 1, 16),
            },
        }

        self.conv_0 = TtConv(
            device,
            parameters,
            "model.0",
            input_params=[3, 2, 1, 80, 16],
            act_block_h=True,
            deallocate_activation=True,
        )
        self.conv_1 = TtConv(
            device,
            parameters,
            "model.1",
            input_params=[3, 2, 1, 160, 80],
            act_block_h=True,
            block_shard=True,
        )
        self.c2f_2 = TtC2f(
            device,
            parameters,
            "model.2",
            n=3,
            shortcut=True,
            input_params=c2f_configs["model.2"]["input_params"],
        )
        self.conv_3 = TtConv(
            device,
            parameters,
            "model.3",
            input_params=[3, 2, 1, 320, 160],
            deallocate_activation=False,
        )
        self.c2f_4 = TtC2f(
            device,
            parameters,
            "model.4",
            n=6,
            shortcut=True,
            input_params=c2f_configs["model.4"]["input_params"],
        )
        self.conv_5 = TtConv(device, parameters, "model.5", input_params=[3, 2, 1, 640, 320], block_shard=True)
        self.c2f_6 = TtC2f(
            device,
            parameters,
            "model.6",
            n=6,
            shortcut=True,
            block_shard=False,
            change_shard=True,
            input_params=c2f_configs["model.6"]["input_params"],
        )
        self.conv_7 = TtConv(device, parameters, "model.7", input_params=[3, 2, 1, 640, 640], block_shard=True)
        self.c2f_8 = TtC2f(
            device,
            parameters,
            "model.8",
            n=3,
            shortcut=True,
            change_shard=True,
            block_shard=True,
            input_params=c2f_configs["model.8"]["input_params"],
        )
        self.sppf_9 = TtSppf(
            device, parameters, "model.9", input_params=sppf_configs["input_params"], batch_size=self.batch_size
        )
        self.c2f_12 = TtC2f(
            device,
            parameters,
            "model.12",
            n=3,
            shortcut=False,
            bfloat8=True,
            block_shard=True,
            input_params=c2f_configs["model.12"]["input_params"],
        )
        self.c2f_15 = TtC2f(
            device,
            parameters,
            "model.15",
            n=3,
            shortcut=False,
            input_params=c2f_configs["model.15"]["input_params"],
        )
        self.conv_16 = TtConv(device, parameters, "model.16", input_params=[3, 2, 1, 320, 320], block_shard=True)
        self.c2f_18 = TtC2f(
            device,
            parameters,
            "model.18",
            n=3,
            shortcut=False,
            input_params=c2f_configs["model.18"]["input_params"],
        )
        self.conv_19 = TtConv(device, parameters, "model.19", input_params=[3, 2, 1, 640, 640], block_shard=True)
        self.c2f_21 = TtC2f(
            device,
            parameters,
            "model.21",
            n=3,
            shortcut=False,
            input_params=c2f_configs["model.21"]["input_params"],
            block_shard=False,
            change_shard=True,
        )
        self.detect_22 = TtDetect(device, parameters, "model.22", detect_config)

    def __call__(self, x):
        conv_0, out_h, out_w = self.conv_0(x)
        conv_1, out_h, out_w = self.conv_1(conv_0)
        ttnn.deallocate(conv_0)
        c2f_2, out_h, out_w = self.c2f_2(conv_1)
        ttnn.deallocate(conv_1)
        conv_3, out_h, out_w = self.conv_3(c2f_2)
        ttnn.deallocate(c2f_2)
        c2f_4, out_h, out_w = self.c2f_4(conv_3)
        ttnn.deallocate(conv_3)
        c2f_4 = ttnn.sharded_to_interleaved(c2f_4, ttnn.L1_MEMORY_CONFIG)
        c2f_4 = ttnn.reallocate(c2f_4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_5, out_h, out_w = self.conv_5(c2f_4)
        c2f_6, out_h, out_w = self.c2f_6(conv_5)
        ttnn.deallocate(conv_5)
        c2f_6 = ttnn.reallocate(c2f_6, memory_config=ttnn.L1_MEMORY_CONFIG)
        conv_7, out_h, out_w = self.conv_7(c2f_6)
        conv_7 = ttnn.sharded_to_interleaved(conv_7, ttnn.L1_MEMORY_CONFIG)
        c2f_8, out_h, out_w = self.c2f_8(conv_7)
        ttnn.deallocate(conv_7)
        c2f_8 = ttnn.sharded_to_interleaved(c2f_8)
        nine, out_h, out_w = self.sppf_9(c2f_8)
        ttnn.deallocate(c2f_8)
        sppf_9 = ttnn.to_layout(nine, ttnn.ROW_MAJOR_LAYOUT)
        sppf_9 = ttnn.reshape(sppf_9, (self.batch_size, out_h, out_w, sppf_9.shape[-1]))
        nhw = sppf_9.shape[0] * sppf_9.shape[1] * sppf_9.shape[2]
        num_cores = determine_num_cores(nhw, sppf_9.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            sppf_9.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if sppf_9.is_sharded():
            sppf_9 = ttnn.reshard(sppf_9, shardspec)
        else:
            sppf_9 = ttnn.interleaved_to_sharded(sppf_9, shardspec)
        sppf_9 = ttnn.upsample(sppf_9, (2, 2), memory_config=sppf_9.memory_config())

        x = ttnn.reshape(sppf_9, (1, 1, (self.batch_size) * sppf_9.shape[1] * sppf_9.shape[2], sppf_9.shape[-1]))

        c2f_6 = ttnn.to_layout(c2f_6, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([x, c2f_6])
        ttnn.deallocate(c2f_6)
        c2f_12, out_h, out_w = self.c2f_12(x)
        ttnn.deallocate(x)
        ttnn.deallocate(sppf_9)
        c2f_12 = ttnn.sharded_to_interleaved(c2f_12, ttnn.L1_MEMORY_CONFIG)
        c2f_12 = ttnn.to_layout(c2f_12, ttnn.ROW_MAJOR_LAYOUT)
        twelve = ttnn.clone(c2f_12, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        c2f_12 = ttnn.reshape(
            c2f_12, (self.batch_size, out_h, out_w, c2f_12.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        nhw = c2f_12.shape[0] * c2f_12.shape[1] * c2f_12.shape[2]
        num_cores = determine_num_cores(nhw, c2f_12.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            c2f_12.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if c2f_12.is_sharded():
            c2f_12 = ttnn.reshard(c2f_12, shardspec)
        else:
            c2f_12 = ttnn.interleaved_to_sharded(c2f_12, shardspec)
        c2f_12 = ttnn.upsample(c2f_12, (2, 2), memory_config=c2f_12.memory_config())

        x = ttnn.reshape(c2f_12, (1, 1, (self.batch_size) * c2f_12.shape[1] * c2f_12.shape[2], c2f_12.shape[-1]))
        c2f_4 = ttnn.to_layout(c2f_4, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([x, c2f_4])
        ttnn.deallocate(c2f_4)
        ttnn.deallocate(c2f_12)

        c2f_15, out_h, out_w = self.c2f_15(x)
        ttnn.deallocate(x)
        c2f_15 = ttnn.sharded_to_interleaved(c2f_15, ttnn.L1_MEMORY_CONFIG)
        fifteen = ttnn.clone(c2f_15, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_16, out_h, out_w = self.conv_16(c2f_15)
        ttnn.deallocate(c2f_15)
        conv_16 = ttnn.sharded_to_interleaved(conv_16, ttnn.L1_MEMORY_CONFIG)
        conv_16 = ttnn.to_layout(conv_16, ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([conv_16, twelve])
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(twelve)
        ttnn.deallocate(conv_16)
        c2f_18, out_h, out_w = self.c2f_18(x)
        ttnn.deallocate(x)
        c2f_18 = ttnn.sharded_to_interleaved(c2f_18, ttnn.L1_MEMORY_CONFIG)
        eighteen = ttnn.clone(c2f_18, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_19, out_h, out_w = self.conv_19(c2f_18)
        ttnn.deallocate(c2f_18)
        conv_19 = ttnn.sharded_to_interleaved(conv_19, ttnn.L1_MEMORY_CONFIG)
        conv_19 = ttnn.to_layout(conv_19, ttnn.ROW_MAJOR_LAYOUT)
        nine = ttnn.to_layout(nine, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([conv_19, nine])
        ttnn.deallocate(nine)
        ttnn.deallocate(conv_19)
        c2f_21, out_h, out_w = self.c2f_21(x)
        ttnn.deallocate(x)
        x = [fifteen, eighteen, c2f_21]
        x = self.detect_22(x, nc=80, ch=(320, 640, 640), reg_max=self.reg_max)
        return x


class TtYolov8xModel:
    def __init__(self, device, parameters, res=(640, 640), batch_size=1):
        self.device = device
        self.parameters = parameters
        self.res = res
        self.detection_model = TtDetectionModel(device, parameters, res, batch_size)

    def __call__(self, x):
        return self.detection_model(x)
