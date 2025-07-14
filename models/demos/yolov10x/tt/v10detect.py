# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.common import Conv, TtYolov10Conv2D, deallocate_tensors
from models.experimental.yolo_common.yolo_utils import concat


class TtnnV10Detect:
    end2end = True

    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.device = device
        self.cv2_0_0 = Conv(
            device,
            parameters.one2one_cv2[0][0],
            self.conv_pt.one2one_cv2[0][0],
            is_detect=True,
        )
        self.cv2_0_1 = Conv(
            device,
            parameters.one2one_cv2[0][1],
            self.conv_pt.one2one_cv2[0][1],
            is_detect=True,
        )
        self.cv2_0_2 = TtYolov10Conv2D(
            parameters.one2one_cv2[0][2], self.conv_pt.one2one_cv2[0][2], device=device, is_detect=True
        )

        self.cv2_1_0 = Conv(device, parameters.one2one_cv2[1][0], self.conv_pt.one2one_cv2[1][0], is_detect=True)
        self.cv2_1_1 = Conv(device, parameters.one2one_cv2[1][1], self.conv_pt.one2one_cv2[1][1], is_detect=True)
        self.cv2_1_2 = TtYolov10Conv2D(
            parameters.one2one_cv2[1][2], self.conv_pt.one2one_cv2[1][2], device=device, is_detect=True
        )

        self.cv2_2_0 = Conv(device, parameters.one2one_cv2[2][0], self.conv_pt.one2one_cv2[2][0], is_detect=True)
        self.cv2_2_1 = Conv(device, parameters.one2one_cv2[2][1], self.conv_pt.one2one_cv2[2][1], is_detect=True)
        self.cv2_2_2 = TtYolov10Conv2D(
            parameters.one2one_cv2[2][2], self.conv_pt.one2one_cv2[2][2], device=device, is_detect=True
        )

        self.cv3_0_0_0 = Conv(
            device,
            parameters.one2one_cv3[0][0][0],
            conv_pt.one2one_cv3[0][0][0],
            is_detect=True,
        )
        self.cv3_0_0_1 = Conv(device, parameters.one2one_cv3[0][0][1], conv_pt.one2one_cv3[0][0][1], is_detect=True)
        self.cv3_0_1_0 = Conv(
            device,
            parameters.one2one_cv3[0][1][0],
            conv_pt.one2one_cv3[0][1][0],
            is_detect=True,
        )
        self.cv3_0_1_1 = Conv(
            device,
            parameters.one2one_cv3[0][1][1],
            conv_pt.one2one_cv3[0][1][1],
            is_detect=True,
        )
        self.cv3_0_2_0 = TtYolov10Conv2D(
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
            device,
            parameters.one2one_cv3[1][0][1],
            conv_pt.one2one_cv3[1][0][1],
            is_detect=True,
        )
        self.cv3_1_1_0 = Conv(device, parameters.one2one_cv3[1][1][0], conv_pt.one2one_cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = Conv(
            device,
            parameters.one2one_cv3[1][1][1],
            conv_pt.one2one_cv3[1][1][1],
            is_detect=True,
        )
        self.cv3_1_2_0 = TtYolov10Conv2D(
            parameters.one2one_cv3[1][2], conv_pt.one2one_cv3[1][2], device=device, is_detect=True
        )

        self.cv3_1_0_0 = Conv(
            device,
            parameters.one2one_cv3[1][0][0],
            conv_pt.one2one_cv3[1][0][0],
            is_detect=True,
            use_1d_systolic_array=False,
        )

        self.cv3_1_0_1 = Conv(device, parameters.one2one_cv3[1][0][1], conv_pt.one2one_cv3[1][0][1], is_detect=True)
        self.cv3_1_1_0 = Conv(device, parameters.one2one_cv3[1][1][0], conv_pt.one2one_cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = Conv(device, parameters.one2one_cv3[1][1][1], conv_pt.one2one_cv3[1][1][1], is_detect=True)
        self.cv3_1_2_0 = TtYolov10Conv2D(
            parameters.one2one_cv3[1][2], conv_pt.one2one_cv3[1][2], device=device, is_detect=True
        )

        self.cv3_2_0_0 = Conv(
            device,
            parameters.one2one_cv3[2][0][0],
            conv_pt.one2one_cv3[2][0][0],
            is_detect=True,
            use_1d_systolic_array=False,
        )

        self.cv3_2_0_1 = Conv(device, parameters.one2one_cv3[2][0][1], conv_pt.one2one_cv3[2][0][1], is_detect=True)
        self.cv3_2_1_0 = Conv(device, parameters.one2one_cv3[2][1][0], conv_pt.one2one_cv3[2][1][0], is_detect=True)
        self.cv3_2_1_1 = Conv(device, parameters.one2one_cv3[2][1][1], conv_pt.one2one_cv3[2][1][1], is_detect=True)
        self.cv3_2_2_0 = TtYolov10Conv2D(
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

        if y2.is_sharded():
            y2 = ttnn.sharded_to_interleaved(y2, ttnn.L1_MEMORY_CONFIG)
        x5 = self.cv3_1_0_0(y2)

        x5 = self.cv3_1_0_1(x5)
        x5 = self.cv3_1_1_0(x5)
        x5 = self.cv3_1_1_1(x5)
        x5 = self.cv3_1_2_0(x5)

        if y3.is_sharded():
            y3 = ttnn.sharded_to_interleaved(y3, ttnn.L1_MEMORY_CONFIG)
        x6 = self.cv3_2_0_0(y3)
        x6 = self.cv3_2_0_1(x6)
        x6 = self.cv3_2_1_0(x6)
        x6 = self.cv3_2_1_1(x6)
        x6 = self.cv3_2_2_0(x6)

        if x1.is_sharded():
            x1 = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        if x2.is_sharded():
            x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        if x3.is_sharded():
            x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        if x4.is_sharded():
            x4 = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        if x5.is_sharded():
            x5 = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        if x6.is_sharded():
            x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1 = concat(-1, False, x1, x4)
        y2 = concat(-1, False, x2, x5)
        y3 = concat(-1, False, x3, x6)

        y = concat(2, False, y1, y2, y3)

        y = ttnn.squeeze(y, dim=0)

        ya, yb = y[:, :, :64], y[:, :, 64:144]
        ya = ttnn.permute(ya, (0, 2, 1))
        ya = ttnn.reshape(ya, (ya.shape[0], 4, 16, y.shape[1]))
        ya = ttnn.permute(ya, (0, 1, 3, 2))
        ya = ttnn.to_layout(ya, ttnn.TILE_LAYOUT)
        ya = ttnn.softmax(ya, dim=-1)
        c = self.dfl(ya)

        deallocate_tensors(y1, y2, y3, x1, x2, x3, x4, x5, x6, y, ya)
        if c.is_sharded():
            c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        if c.get_layout() == ttnn.TILE_LAYOUT:
            c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]
        anchor, strides = self.anchors, self.strides
        if c1.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        if c2.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)
        c1 = anchor - c1
        c2 = anchor + c2

        z = concat(1, False, c1, c2)

        z = ttnn.multiply(z, strides)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid_accurate(yb)

        output = concat(1, False, z, yb)

        deallocate_tensors(c, c1, c2, yb, z)

        return output
