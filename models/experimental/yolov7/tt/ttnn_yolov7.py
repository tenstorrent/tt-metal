# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolo_common.yolo_utils import concat, determine_num_cores, get_core_grid_from_num_cores
from models.experimental.yolov7.tt.common import TtYOLOv7Conv2D as Conv


class ttnn_SPPCSPC:
    def __init__(self, device, parameters, k=(5, 9, 13)) -> None:
        self.device = device
        self.parameters = parameters
        self.k = k
        self.cv1 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["cv1"],
        )
        self.cv2 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["cv2"],
        )
        self.cv3 = Conv(
            [1, 20, 20, 512],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["cv3"],
            height_sharding=False,
        )
        self.cv4 = Conv(
            [1, 20, 20, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["cv4"],
        )
        self.cv5 = Conv(
            [1, 20, 20, 2048],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["cv5"],
            height_sharding=False,
        )
        self.cv6 = Conv(
            [1, 20, 20, 512],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["cv6"],
            height_sharding=False,
        )
        self.cv7 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["cv7"],
        )

    def __call__(self, x):
        x1 = self.cv1(self.device, x)
        x1 = self.cv3(self.device, x1)
        x1 = self.cv4(self.device, x1)
        x1 = ttnn.sharded_to_interleaved(x1, ttnn.L1_MEMORY_CONFIG)
        x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x1_m1 = ttnn.max_pool2d(
            input_tensor=x1,
            batch_size=1,
            input_h=20,
            input_w=20,
            channels=512,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x1_m2 = ttnn.max_pool2d(
            input_tensor=x1,
            batch_size=1,
            input_h=20,
            input_w=20,
            channels=512,
            kernel_size=[9, 9],
            stride=[1, 1],
            padding=[4, 4],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        x1_m3 = ttnn.max_pool2d(
            input_tensor=x1,
            batch_size=1,
            input_h=20,
            input_w=20,
            channels=512,
            kernel_size=[13, 13],
            stride=[1, 1],
            padding=[6, 6],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x1 = ttnn.to_layout(x1, ttnn.ROW_MAJOR_LAYOUT)

        x1_m1 = ttnn.sharded_to_interleaved(x1_m1, ttnn.L1_MEMORY_CONFIG)

        x1_m2 = ttnn.sharded_to_interleaved(x1_m2, ttnn.L1_MEMORY_CONFIG)

        x1_m3 = ttnn.sharded_to_interleaved(x1_m3, ttnn.L1_MEMORY_CONFIG)

        y1 = concat(3, False, x1, x1_m1, x1_m2, x1_m3)
        ttnn.deallocate(x1)
        ttnn.deallocate(x1_m1)
        ttnn.deallocate(x1_m2)
        ttnn.deallocate(x1_m3)

        y1 = self.cv5(self.device, y1)

        y1 = self.cv6(self.device, y1)

        y2 = self.cv2(self.device, x)

        y1 = ttnn.sharded_to_interleaved(y1, ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.sharded_to_interleaved(y2, ttnn.L1_MEMORY_CONFIG)

        out = concat(3, False, y1, y2)

        out = self.cv7(self.device, out)

        ttnn.deallocate(y1)
        ttnn.deallocate(y2)

        return out


class ttnn_repconv:
    def __init__(self, device, parameters, input_shape) -> None:
        self.device = device
        self.parameters = parameters
        self.rbr_dense = Conv(
            input_shape,
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["0"],
            height_sharding=False,
        )
        self.rbr_1x1 = Conv(
            input_shape,
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["1"],
            height_sharding=False,
        )

    def __call__(self, x):
        x1 = self.rbr_dense(self.device, x)
        x2 = self.rbr_1x1(self.device, x)
        out = ttnn.add(x1, x2)
        out = ttnn.silu(out)
        ttnn.deallocate(x)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        return out


class ttnn_detect:
    stride = None
    export = False
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, device, parameters, grid_tensors, nc=80, anchors=(), ch=()) -> None:
        self.device = device
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [ttnn.zeros([1])] * self.nl
        self.grid_tensors = grid_tensors
        self.grid[0] = ttnn.from_torch(grid_tensors[0], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.grid[1] = ttnn.from_torch(grid_tensors[1], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.grid[2] = ttnn.from_torch(grid_tensors[2], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        a = ttnn.from_torch(a, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.anchors = ttnn.clone(a)

        self.anchor_grid = ttnn.reshape((ttnn.clone(a)), (self.nl, -1, 1, 1, 2))
        self.stride = [8.0, 16.0, 32.0]

        self.m = []
        self.convm_1 = Conv([1, 80, 80, 256], (1, 1, 1, 1, 0, 0, 1, 1), parameters["0"], is_reshape=True, activation="")
        self.m.append(self.convm_1)

        self.convm_2 = Conv([1, 40, 40, 512], (1, 1, 1, 1, 0, 0, 1, 1), parameters["1"], is_reshape=True, activation="")
        self.m.append(self.convm_2)

        self.convm_2 = Conv(
            [1, 20, 20, 1024], (1, 1, 1, 1, 0, 0, 1, 1), parameters["2"], is_reshape=True, activation=""
        )
        self.m.append(self.convm_2)

    def __call__(self, x):
        z = []
        self.training = False

        for i in range(self.nl):
            x[i] = ttnn.to_memory_config(x[i], ttnn.L1_MEMORY_CONFIG)
            x[i] = self.m[i](self.device, x[i])
            bs, _, ny, nx = x[i].shape

            x[i] = ttnn.reshape(x[i], (bs, self.na, self.no, ny, nx))
            x[i] = ttnn.permute(x[i], (0, 1, 3, 4, 2))

            if not self.training:
                x[i] = ttnn.to_layout(x[i], ttnn.TILE_LAYOUT)
                y = ttnn.sigmoid(x[i], memory_config=ttnn.L1_MEMORY_CONFIG)

                if not torch.onnx.is_in_onnx_export():
                    y = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
                    y = ttnn.permute(y, (0, 1, 4, 2, 3))
                    c1 = y[:, :, 0:2, :, :]
                    d1 = y[:, :, 2:4, :, :]
                    e1 = y[:, :, 4:, :, :]
                    self.grid[i] = ttnn.to_memory_config(self.grid[i], memory_config=ttnn.L1_MEMORY_CONFIG)
                    self.anchor_grid = ttnn.to_memory_config(self.anchor_grid, memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn_grid = ttnn.permute(self.grid[i], (0, 1, 4, 2, 3))
                    ttnn_grid = concat(1, False, ttnn_grid, ttnn_grid, ttnn_grid)

                    c2 = c1 * 2 - 0.5 + ttnn_grid
                    c3 = c2 * self.stride[i]

                    ttnn_anchor_grid = ttnn.permute(self.anchor_grid[i : i + 1], (0, 1, 4, 2, 3))

                    d2 = ttnn.pow((d1 * 2), 2) * ttnn_anchor_grid
                    y_cat = concat(2, False, c3, d2, e1)

                    y_cat = ttnn.to_memory_config(y_cat, memory_config=ttnn.L1_MEMORY_CONFIG)
                    y_cat = ttnn.permute(y_cat, (0, 1, 3, 4, 2))
                z.append(ttnn.reshape(y_cat, (bs, -1, self.no)))

        out = (ttnn.concat(z, 1), x)
        return out


class ttnn_yolov7:
    def __init__(self, device, parameters, grid_tensors) -> None:
        self.device = device
        self.parameters = parameters
        self.nc = 80
        self.anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
        self.ch = [256, 512, 1024]
        self.grid_tensors = grid_tensors
        self.conv1 = Conv([1, 640, 640, 3], (3, 3, 1, 1, 1, 1, 1, 1), parameters["0"], act_block_h=64)
        self.conv2 = Conv([1, 640, 640, 32], (3, 3, 2, 2, 1, 1, 1, 1), parameters["1"])
        self.conv3 = Conv(
            [1, 320, 320, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["2"],
            act_block_h=64,
        )
        self.conv4 = Conv(
            [1, 320, 320, 64],
            (3, 3, 2, 2, 1, 1, 1, 1),
            parameters["3"],
        )
        self.conv5 = Conv(
            [1, 160, 160, 128],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["4"],
        )
        self.conv6 = Conv(
            [1, 160, 160, 128],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["5"],
        )
        self.conv7 = Conv(
            [1, 160, 160, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["6"],
        )
        self.conv8 = Conv(
            [1, 160, 160, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["7"],
        )
        self.conv9 = Conv(
            [1, 160, 160, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["8"],
        )
        self.conv10 = Conv(
            [1, 160, 160, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["9"],
        )

        self.conv11 = Conv(
            [1, 160, 160, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["11"],
        )
        self.conv12 = Conv(
            [1, 80, 80, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["13"],
            height_sharding=False,
        )
        self.conv13 = Conv(
            [1, 160, 160, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["14"],
        )
        self.conv14 = Conv(
            [1, 160, 160, 128],
            (3, 3, 2, 2, 1, 1, 1, 1),
            parameters["15"],
        )

        self.conv15 = Conv(
            [1, 80, 80, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["17"],
        )
        self.conv16 = Conv(
            [1, 80, 80, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["18"],
        )
        self.conv17 = Conv(
            [1, 80, 80, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["19"],
        )
        self.conv18 = Conv(
            [1, 80, 80, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["20"],
        )
        self.conv19 = Conv(
            [1, 80, 80, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["21"],
        )
        self.conv20 = Conv(
            [1, 80, 80, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["22"],
        )

        self.conv21 = Conv(
            [1, 80, 80, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["24"],
        )
        self.conv22 = Conv(
            [1, 40, 40, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["26"],
        )
        self.conv23 = Conv(
            [1, 80, 80, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["27"],
            height_sharding=False,
        )
        self.conv24 = Conv(
            [1, 80, 80, 256],
            (3, 3, 2, 2, 1, 1, 1, 1),
            parameters["28"],
        )

        self.conv25 = Conv(
            [1, 40, 40, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["30"],
        )
        self.conv26 = Conv(
            [1, 40, 40, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["31"],
        )
        self.conv27 = Conv(
            [1, 40, 40, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["32"],
        )
        self.conv28 = Conv(
            [1, 40, 40, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["33"],
        )
        self.conv29 = Conv(
            [1, 40, 40, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["34"],
        )
        self.conv30 = Conv(
            [1, 40, 40, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["35"],
        )

        self.conv31 = Conv(
            [1, 40, 40, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["37"],
            height_sharding=False,
        )
        self.conv32 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["39"],
        )
        self.conv33 = Conv(
            [1, 40, 40, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["40"],
        )
        self.conv34 = Conv(
            [1, 40, 40, 512],
            (3, 3, 2, 2, 1, 1, 1, 1),
            parameters["41"],
            act_block_h=64,
            height_sharding=False,
            enable_act_double_buffer=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.conv35 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["43"],
        )
        self.conv36 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["44"],
        )
        self.conv37 = Conv(
            [1, 20, 20, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["45"],
        )
        self.conv38 = Conv(
            [1, 20, 20, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["46"],
        )
        self.conv39 = Conv(
            [1, 20, 20, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["47"],
        )
        self.conv40 = Conv(
            [1, 20, 20, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["48"],
        )

        self.conv41 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["50"],
            height_sharding=False,
        )
        self.SPPCSPC = ttnn_SPPCSPC(device, parameters["51"])

        self.conv42 = Conv(
            [1, 20, 20, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["52"],
        )
        self.conv43 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["54"],
            num_cores_nhw=56,
        )

        self.conv44 = Conv(
            [1, 40, 40, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["56"],
        )
        self.conv45 = Conv(
            [1, 40, 40, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["57"],
        )
        self.conv46 = Conv(
            [1, 40, 40, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["58"],
        )
        self.conv47 = Conv(
            [1, 40, 40, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["59"],
        )
        self.conv48 = Conv(
            [1, 40, 40, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["60"],
        )
        self.conv49 = Conv(
            [1, 40, 40, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["61"],
        )

        self.conv50 = Conv(
            [1, 40, 40, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["63"],
        )
        self.conv51 = Conv(
            [1, 40, 40, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["64"],
        )
        self.conv52 = Conv(
            [1, 80, 80, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["66"],
        )

        self.conv53 = Conv(
            [1, 80, 80, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["68"],
        )
        self.conv54 = Conv(
            [1, 80, 80, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["69"],
        )
        self.conv55 = Conv(
            [1, 80, 80, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["70"],
        )
        self.conv56 = Conv(
            [1, 80, 80, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["71"],
        )
        self.conv57 = Conv(
            [1, 80, 80, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["72"],
        )
        self.conv58 = Conv(
            [1, 80, 80, 64],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["73"],
        )

        self.conv59 = Conv(
            [1, 80, 80, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["75"],
        )
        self.conv60 = Conv(
            [1, 40, 40, 128],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["77"],
        )
        self.conv61 = Conv(
            [1, 80, 80, 128],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["78"],
            input_channels_alignment=32,
        )
        self.conv62 = Conv(
            [1, 80, 80, 128],
            (3, 3, 2, 2, 1, 1, 1, 1),
            parameters["79"],
        )

        self.conv63 = Conv(
            [1, 40, 40, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["81"],
        )
        self.conv64 = Conv(
            [1, 40, 40, 512],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["82"],
        )
        self.conv65 = Conv(
            [1, 40, 40, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["83"],
        )
        self.conv66 = Conv(
            [1, 40, 40, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["84"],
        )
        self.conv67 = Conv(
            [1, 40, 40, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["85"],
        )
        self.conv68 = Conv(
            [1, 40, 40, 128],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["86"],
        )

        self.conv69 = Conv(
            [1, 40, 40, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["88"],
        )
        self.conv70 = Conv(
            [1, 20, 20, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["90"],
        )
        self.conv71 = Conv(
            [1, 40, 40, 256],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["91"],
        )
        self.conv72 = Conv(
            [1, 40, 40, 256],
            (3, 3, 2, 2, 1, 1, 1, 1),
            parameters["92"],
        )

        self.conv73 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["94"],
            height_sharding=False,
        )
        self.conv74 = Conv(
            [1, 20, 20, 1024],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["95"],
            height_sharding=False,
        )
        self.conv75 = Conv(
            [1, 20, 20, 512],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["96"],
        )
        self.conv76 = Conv(
            [1, 20, 20, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["97"],
        )
        self.conv77 = Conv(
            [1, 20, 20, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["98"],
        )
        self.conv78 = Conv(
            [1, 20, 20, 256],
            (3, 3, 1, 1, 1, 1, 1, 1),
            parameters["99"],
        )

        self.conv79 = Conv(
            [1, 20, 20, 2048],
            (1, 1, 1, 1, 0, 0, 1, 1),
            parameters["101"],
            height_sharding=False,
        )
        self.repconv1 = ttnn_repconv(device, parameters["102"], [1, 80, 80, 128])
        self.repconv2 = ttnn_repconv(device, parameters["103"], [1, 40, 40, 256])
        self.repconv3 = ttnn_repconv(device, parameters["104"], [1, 20, 20, 512])
        self.detect = ttnn_detect(device, parameters["105"], self.grid_tensors, self.nc, self.anchors, self.ch)

    def __call__(self, input_tensor):
        conv1 = self.conv1(self.device, input_tensor)

        conv2 = self.conv2(self.device, conv1)
        ttnn.deallocate(conv1)

        conv3 = self.conv3(self.device, conv2)
        ttnn.deallocate(conv2)

        conv4 = self.conv4(self.device, conv3)
        ttnn.deallocate(conv3)

        conv5 = self.conv5(self.device, conv4)

        conv6 = self.conv6(self.device, conv4)

        conv7 = self.conv7(self.device, conv6)

        conv8 = self.conv8(self.device, conv7)

        conv9 = self.conv9(self.device, conv8)

        conv10 = self.conv10(self.device, conv9)

        conv10 = ttnn.reshape(conv10, (1, 160, 160, 64))

        conv8 = ttnn.reshape(conv8, (1, 160, 160, 64))

        conv6 = ttnn.reshape(conv6, (1, 160, 160, 64))

        conv5 = ttnn.reshape(conv5, (1, 160, 160, 64))

        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [416, 256],
            core_grid=conv8.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        conv10 = concat(3, False, conv10, conv8, conv6, conv5)
        ttnn.deallocate(conv4)
        ttnn.deallocate(conv7)
        ttnn.deallocate(conv9)

        conv11 = self.conv11(self.device, conv10)
        ttnn.deallocate(conv5)
        ttnn.deallocate(conv6)
        ttnn.deallocate(conv8)

        mp1 = ttnn.max_pool2d(
            input_tensor=conv11,
            batch_size=1,
            input_h=160,
            input_w=160,
            channels=256,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(conv10)

        mp1 = ttnn.sharded_to_interleaved(mp1, ttnn.L1_MEMORY_CONFIG)
        conv12 = self.conv12(self.device, mp1)

        conv13 = self.conv13(self.device, conv11)

        conv14 = self.conv14(self.device, conv13)

        conv14 = ttnn.to_layout(conv14, ttnn.ROW_MAJOR_LAYOUT)
        conv14 = ttnn.sharded_to_interleaved(conv14, ttnn.L1_MEMORY_CONFIG)

        conv12 = ttnn.to_layout(conv12, ttnn.ROW_MAJOR_LAYOUT)
        conv12 = ttnn.sharded_to_interleaved(conv12, ttnn.L1_MEMORY_CONFIG)

        conv14 = concat(3, True, conv14, conv12)
        ttnn.deallocate(conv11)
        ttnn.deallocate(mp1)
        ttnn.deallocate(conv13)

        conv15 = self.conv15(self.device, conv14)
        ttnn.deallocate(conv12)

        conv16 = self.conv16(self.device, conv14)
        ttnn.deallocate(conv14)

        conv17 = self.conv17(self.device, conv16)

        conv18 = self.conv18(self.device, conv17)

        conv19 = self.conv19(self.device, conv18)

        conv20 = self.conv20(self.device, conv19)

        conv20 = ttnn.to_layout(conv20, ttnn.ROW_MAJOR_LAYOUT)
        conv20 = ttnn.sharded_to_interleaved(conv20, ttnn.L1_MEMORY_CONFIG)

        conv18 = ttnn.to_layout(conv18, ttnn.ROW_MAJOR_LAYOUT)
        conv18 = ttnn.sharded_to_interleaved(conv18, ttnn.L1_MEMORY_CONFIG)

        conv16 = ttnn.to_layout(conv16, ttnn.ROW_MAJOR_LAYOUT)
        conv16 = ttnn.sharded_to_interleaved(conv16, ttnn.L1_MEMORY_CONFIG)

        conv15 = ttnn.to_layout(conv15, ttnn.ROW_MAJOR_LAYOUT)
        conv15 = ttnn.sharded_to_interleaved(conv15, ttnn.L1_MEMORY_CONFIG)

        conv20 = concat(3, True, conv20, conv18, conv16, conv15)
        ttnn.deallocate(conv17)
        ttnn.deallocate(conv19)

        conv21 = self.conv21(self.device, conv20)
        ttnn.deallocate(conv15)
        ttnn.deallocate(conv16)
        ttnn.deallocate(conv18)

        mp2 = ttnn.max_pool2d(
            input_tensor=conv21,
            batch_size=1,
            input_h=80,
            input_w=80,
            channels=512,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(conv20)

        conv22 = self.conv22(self.device, mp2)

        conv23 = self.conv23(self.device, conv21)

        conv24 = self.conv24(self.device, conv23)

        conv24 = ttnn.sharded_to_interleaved(conv24, ttnn.L1_MEMORY_CONFIG)

        conv22 = ttnn.sharded_to_interleaved(conv22, ttnn.L1_MEMORY_CONFIG)

        conv24 = concat(3, True, conv24, conv22)
        ttnn.deallocate(mp2)
        ttnn.deallocate(conv23)

        conv25 = self.conv25(self.device, conv24)
        ttnn.deallocate(conv22)

        conv26 = self.conv26(self.device, conv24)
        ttnn.deallocate(conv24)

        conv27 = self.conv27(self.device, conv26)

        conv28 = self.conv28(self.device, conv27)

        conv29 = self.conv29(self.device, conv28)

        conv30 = self.conv30(self.device, conv29)

        conv30 = ttnn.sharded_to_interleaved(conv30, ttnn.L1_MEMORY_CONFIG)

        conv28 = ttnn.sharded_to_interleaved(conv28, ttnn.L1_MEMORY_CONFIG)

        conv26 = ttnn.sharded_to_interleaved(conv26, ttnn.L1_MEMORY_CONFIG)

        conv25 = ttnn.sharded_to_interleaved(conv25, ttnn.L1_MEMORY_CONFIG)

        conv30 = concat(3, False, conv30, conv28, conv26, conv25)
        ttnn.deallocate(conv27)
        ttnn.deallocate(conv29)

        conv31 = self.conv31(self.device, conv30)
        ttnn.deallocate(conv25)
        ttnn.deallocate(conv26)
        ttnn.deallocate(conv28)

        conv31 = ttnn.sharded_to_interleaved(conv31, ttnn.L1_MEMORY_CONFIG)
        conv31 = ttnn.to_layout(conv31, ttnn.ROW_MAJOR_LAYOUT)
        mp3 = ttnn.max_pool2d(
            input_tensor=conv31,
            batch_size=1,
            input_h=40,
            input_w=40,
            channels=1024,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(conv30)

        mp3 = ttnn.sharded_to_interleaved(mp3, ttnn.L1_MEMORY_CONFIG)
        mp3 = ttnn.to_layout(mp3, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        conv32 = self.conv32(self.device, mp3)

        conv33 = self.conv33(self.device, conv31)

        conv34 = self.conv34(self.device, conv33)

        conv34 = ttnn.sharded_to_interleaved(conv34, ttnn.L1_MEMORY_CONFIG)
        conv32 = ttnn.sharded_to_interleaved(conv32, ttnn.L1_MEMORY_CONFIG)

        conv34 = concat(3, False, conv34, conv32)
        ttnn.deallocate(mp3)
        ttnn.deallocate(conv33)

        conv35 = self.conv35(self.device, conv34)
        ttnn.deallocate(conv32)

        conv36 = self.conv36(self.device, conv34)
        ttnn.deallocate(conv34)

        conv37 = self.conv37(self.device, conv36)

        conv38 = self.conv38(self.device, conv37)

        conv39 = self.conv39(self.device, conv38)

        conv40 = self.conv40(self.device, conv39)

        conv40 = ttnn.sharded_to_interleaved(conv40, ttnn.L1_MEMORY_CONFIG)
        conv38 = ttnn.sharded_to_interleaved(conv38, ttnn.L1_MEMORY_CONFIG)
        conv36 = ttnn.sharded_to_interleaved(conv36, ttnn.L1_MEMORY_CONFIG)
        conv35 = ttnn.sharded_to_interleaved(conv35, ttnn.L1_MEMORY_CONFIG)

        conv40 = concat(3, False, conv40, conv38, conv36, conv35)
        ttnn.deallocate(conv37)
        ttnn.deallocate(conv39)

        conv41 = self.conv41(self.device, conv40)
        ttnn.deallocate(conv35)
        ttnn.deallocate(conv36)
        ttnn.deallocate(conv38)
        ttnn.deallocate(conv40)

        SPPCSPC = self.SPPCSPC(conv41)

        conv42 = self.conv42(self.device, SPPCSPC)

        conv42 = ttnn.sharded_to_interleaved(conv42, ttnn.L1_MEMORY_CONFIG)
        conv42 = ttnn.to_layout(conv42, ttnn.ROW_MAJOR_LAYOUT)

        conv42 = ttnn.reshape(conv42, (1, 20, 20, 256))
        x = conv42
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)

        conv42 = x
        conv42 = ttnn.upsample(conv42, 2, memory_config=conv42.memory_config())
        conv42 = ttnn.reshape(conv42, (1, 1, 1600, 256))

        conv43 = self.conv43(self.device, conv31)

        conv43 = ttnn.sharded_to_interleaved(conv43, ttnn.L1_MEMORY_CONFIG)
        conv43 = ttnn.to_layout(conv43, ttnn.ROW_MAJOR_LAYOUT)

        conv42 = ttnn.sharded_to_interleaved(conv42, ttnn.L1_MEMORY_CONFIG)

        conv43 = concat(3, True, conv43, conv42)

        ttnn.deallocate(conv31)

        conv44 = self.conv44(self.device, conv43)
        ttnn.deallocate(conv42)

        conv45 = self.conv45(self.device, conv43)

        conv46 = self.conv46(self.device, conv45)
        ttnn.deallocate(conv43)

        conv47 = self.conv47(self.device, conv46)

        conv48 = self.conv48(self.device, conv47)

        conv49 = self.conv49(self.device, conv48)

        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [32, 1024],
            core_grid=conv48.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        conv49 = concat(3, False, conv49, conv48, conv47, conv46, conv45, conv44)

        conv50 = self.conv50(self.device, conv49)
        ttnn.deallocate(conv44)
        ttnn.deallocate(conv45)
        ttnn.deallocate(conv46)
        ttnn.deallocate(conv47)
        ttnn.deallocate(conv48)

        conv51 = self.conv51(self.device, conv50)
        ttnn.deallocate(conv49)

        conv51 = ttnn.sharded_to_interleaved(conv51, ttnn.L1_MEMORY_CONFIG)
        conv51 = ttnn.to_layout(conv51, ttnn.ROW_MAJOR_LAYOUT)
        conv51 = ttnn.reshape(conv51, (1, 40, 40, 128))

        x = conv51
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)

        conv51 = x

        conv51 = ttnn.upsample(conv51, 2, memory_config=conv51.memory_config())
        conv52 = self.conv52(self.device, conv21)
        conv52 = ttnn.to_layout(conv52, ttnn.ROW_MAJOR_LAYOUT)
        conv52 = ttnn.reshape(conv52, (1, 80, 80, 128))
        conv52 = ttnn.sharded_to_interleaved(conv52, ttnn.L1_MEMORY_CONFIG)

        conv51 = ttnn.sharded_to_interleaved(conv51, ttnn.L1_MEMORY_CONFIG)
        conv52 = concat(3, False, conv52, conv51)

        ttnn.deallocate(conv51)

        conv53 = self.conv53(self.device, conv52)

        conv54 = self.conv54(self.device, conv52)

        conv55 = self.conv55(self.device, conv54)

        conv56 = self.conv56(self.device, conv55)

        conv57 = self.conv57(self.device, conv56)

        conv58 = self.conv58(self.device, conv57)

        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [128, 512],
            core_grid=conv57.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        conv58 = ttnn.concat(
            [conv58, conv57, conv56, conv55, conv54, conv53], dim=3, memory_config=output_sharded_memory_config
        )

        ttnn.deallocate(conv52)

        conv59 = self.conv59(self.device, conv58)
        ttnn.deallocate(conv53)
        ttnn.deallocate(conv54)
        ttnn.deallocate(conv55)
        ttnn.deallocate(conv56)
        ttnn.deallocate(conv57)
        ttnn.deallocate(conv58)

        mp4 = ttnn.max_pool2d(
            input_tensor=conv59,
            batch_size=1,
            input_h=80,
            input_w=80,
            channels=128,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        conv60 = self.conv60(self.device, mp4)

        conv61 = self.conv61(self.device, conv59)

        conv62 = self.conv62(self.device, conv61)

        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [32, 512],
            core_grid=conv60.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        conv62 = ttnn.concat([conv62, conv60, conv50], dim=3, memory_config=output_sharded_memory_config)

        ttnn.deallocate(conv50)
        ttnn.deallocate(conv60)
        ttnn.deallocate(conv61)

        conv63 = self.conv63(self.device, conv62)

        conv64 = self.conv64(self.device, conv62)

        conv65 = self.conv65(self.device, conv64)
        ttnn.deallocate(conv62)

        conv66 = self.conv66(self.device, conv65)

        conv67 = self.conv67(self.device, conv66)

        conv68 = self.conv68(self.device, conv67)

        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [32, 1024],
            core_grid=conv67.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        conv68 = ttnn.concat(
            [conv68, conv67, conv66, conv65, conv64, conv63],
            dim=3,
            memory_config=output_sharded_memory_config,
        )

        conv69 = self.conv69(self.device, conv68)
        ttnn.deallocate(conv63)
        ttnn.deallocate(conv64)
        ttnn.deallocate(conv65)
        ttnn.deallocate(conv66)
        ttnn.deallocate(conv67)

        mp5 = ttnn.max_pool2d(
            input_tensor=conv69,
            batch_size=1,
            input_h=40,
            input_w=40,
            channels=256,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(conv68)

        mp5 = ttnn.to_layout(mp5, ttnn.ROW_MAJOR_LAYOUT)
        mp5 = ttnn.sharded_to_interleaved(mp5, ttnn.L1_MEMORY_CONFIG)
        conv70 = self.conv70(self.device, mp5)

        conv71 = self.conv71(self.device, conv69)

        conv71 = ttnn.sharded_to_interleaved(conv71, ttnn.L1_MEMORY_CONFIG)
        conv72 = self.conv72(self.device, conv71)
        ttnn.deallocate(mp5)

        conv72 = ttnn.sharded_to_interleaved(conv72, ttnn.L1_MEMORY_CONFIG)
        conv72 = ttnn.to_layout(conv72, ttnn.ROW_MAJOR_LAYOUT)

        conv70 = ttnn.sharded_to_interleaved(conv70, ttnn.L1_MEMORY_CONFIG)
        conv70 = ttnn.to_layout(conv70, ttnn.ROW_MAJOR_LAYOUT)

        SPPCSPC = ttnn.sharded_to_interleaved(SPPCSPC, ttnn.L1_MEMORY_CONFIG)
        SPPCSPC = ttnn.to_layout(SPPCSPC, ttnn.ROW_MAJOR_LAYOUT)

        conv72 = concat(3, False, conv72, conv70, SPPCSPC)

        ttnn.deallocate(conv71)

        conv73 = self.conv73(self.device, conv72)
        ttnn.deallocate(conv70)
        ttnn.deallocate(SPPCSPC)

        conv74 = self.conv74(self.device, conv72)

        conv75 = self.conv75(self.device, conv74)
        ttnn.deallocate(conv72)

        conv76 = self.conv76(self.device, conv75)

        conv77 = self.conv77(self.device, conv76)

        conv78 = self.conv78(self.device, conv77)

        conv73 = ttnn.sharded_to_interleaved(conv73, ttnn.L1_MEMORY_CONFIG)
        conv74 = ttnn.sharded_to_interleaved(conv74, ttnn.L1_MEMORY_CONFIG)
        conv75 = ttnn.sharded_to_interleaved(conv75, ttnn.L1_MEMORY_CONFIG)
        conv76 = ttnn.sharded_to_interleaved(conv76, ttnn.L1_MEMORY_CONFIG)
        conv77 = ttnn.sharded_to_interleaved(conv77, ttnn.L1_MEMORY_CONFIG)
        conv78 = ttnn.sharded_to_interleaved(conv78, ttnn.L1_MEMORY_CONFIG)

        conv78 = concat(3, False, conv78, conv77, conv76, conv75, conv74, conv73)

        conv79 = self.conv79(self.device, conv78)
        ttnn.deallocate(conv73)
        ttnn.deallocate(conv74)
        ttnn.deallocate(conv75)
        ttnn.deallocate(conv76)
        ttnn.deallocate(conv77)

        repconv1 = self.repconv1(conv59)
        repconv2 = self.repconv2(conv69)
        repconv3 = self.repconv3(conv79)

        output = self.detect([repconv1, repconv2, repconv3])

        return output
