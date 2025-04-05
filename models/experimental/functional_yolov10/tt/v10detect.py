# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov10.tt.common import Conv, TtYolov10_Conv2D


class TtnnV10Detect:
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
