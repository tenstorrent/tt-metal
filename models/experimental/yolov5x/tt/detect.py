# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.yolo_common.yolo_utils import concat
from models.experimental.yolov5x.tt.common import TtYOLOv5xConv2D


class TtnnDetect:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.device = device

        self.cv2_0_0 = TtYOLOv5xConv2D(
            device, parameters.cv2[0][0].conv, self.conv_pt.cv2[0][0].conv, is_detect=True, activation="silu"
        )
        self.cv2_0_1 = TtYOLOv5xConv2D(
            device, parameters.cv2[0][1].conv, self.conv_pt.cv2[0][1].conv, is_detect=True, activation="silu"
        )
        self.cv2_0_2 = TtYOLOv5xConv2D(device, parameters.cv2[0][2], self.conv_pt.cv2[0][2], is_detect=True)

        self.cv2_1_0 = TtYOLOv5xConv2D(
            device, parameters.cv2[1][0].conv, self.conv_pt.cv2[1][0].conv, is_detect=True, activation="silu"
        )
        self.cv2_1_1 = TtYOLOv5xConv2D(
            device, parameters.cv2[1][1].conv, self.conv_pt.cv2[1][1].conv, is_detect=True, activation="silu"
        )
        self.cv2_1_2 = TtYOLOv5xConv2D(device, parameters.cv2[1][2], self.conv_pt.cv2[1][2], is_detect=True)

        self.cv2_2_0 = TtYOLOv5xConv2D(
            device, parameters.cv2[2][0].conv, self.conv_pt.cv2[2][0].conv, is_detect=True, activation="silu"
        )
        self.cv2_2_1 = TtYOLOv5xConv2D(
            device, parameters.cv2[2][1].conv, self.conv_pt.cv2[2][1].conv, is_detect=True, activation="silu"
        )
        self.cv2_2_2 = TtYOLOv5xConv2D(device, parameters.cv2[2][2], self.conv_pt.cv2[2][2], is_detect=True)

        self.cv3_0_0 = TtYOLOv5xConv2D(
            device, parameters.cv3[0][0].conv, self.conv_pt.cv3[0][0].conv, is_detect=True, activation="silu"
        )
        self.cv3_0_1 = TtYOLOv5xConv2D(
            device, parameters.cv3[0][1].conv, self.conv_pt.cv3[0][1].conv, is_detect=True, activation="silu"
        )
        self.cv3_0_2 = TtYOLOv5xConv2D(device, parameters.cv3[0][2], self.conv_pt.cv3[0][2], is_detect=True)

        self.cv3_1_0 = TtYOLOv5xConv2D(
            device, parameters.cv3[1][0].conv, self.conv_pt.cv3[1][0].conv, is_detect=True, activation="silu"
        )
        self.cv3_1_1 = TtYOLOv5xConv2D(
            device, parameters.cv3[1][1].conv, self.conv_pt.cv3[1][1].conv, is_detect=True, activation="silu"
        )
        self.cv3_1_2 = TtYOLOv5xConv2D(device, parameters.cv3[1][2], self.conv_pt.cv3[1][2], is_detect=True)

        self.cv3_2_0 = TtYOLOv5xConv2D(
            device,
            parameters.cv3[2][0].conv,
            self.conv_pt.cv3[2][0].conv,
            is_detect=True,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.cv3_2_1 = TtYOLOv5xConv2D(
            device, parameters.cv3[2][1].conv, self.conv_pt.cv3[2][1].conv, is_detect=True, activation="silu"
        )
        self.cv3_2_2 = TtYOLOv5xConv2D(device, parameters.cv3[2][2], self.conv_pt.cv3[2][2], is_detect=True)

        self.dfl = TtYOLOv5xConv2D(device, parameters.dfl.conv, self.conv_pt.dfl.conv, is_dfl=True)

        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, y1, y2, y3):
        x1 = self.cv2_0_0(y1)
        x1 = self.cv2_0_1(x1)
        x1 = self.cv2_0_2(x1)
        x1 = ttnn.to_memory_config(x1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x2 = self.cv2_1_0(y2)
        x2 = self.cv2_1_1(x2)
        x2 = self.cv2_1_2(x2)
        x3 = self.cv2_2_0(y3)
        x3 = self.cv2_2_1(x3)
        x3 = self.cv2_2_2(x3)
        x4 = self.cv3_0_0(y1)
        x4 = self.cv3_0_1(x4)
        x4 = self.cv3_0_2(x4)
        x1 = ttnn.to_memory_config(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y1 = concat(-1, False, x1, x4)
        y1 = ttnn.to_memory_config(y1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x1)
        ttnn.deallocate(x4)
        x5 = self.cv3_1_0(y2)
        x5 = self.cv3_1_1(x5)
        x5 = self.cv3_1_2(x5)
        y2 = concat(-1, False, x2, x5)
        ttnn.deallocate(x2)
        ttnn.deallocate(x5)
        x6 = self.cv3_2_0(y3)
        x6 = self.cv3_2_1(x6)
        x6 = self.cv3_2_2(x6)

        y3 = concat(-1, False, x3, x6)
        ttnn.deallocate(x3)
        ttnn.deallocate(x6)
        y1 = ttnn.to_memory_config(y1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = concat(2, False, y1, y2, y3)

        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)

        ya, yb = y[:, :, :, :64], y[:, :, :, 64:144]
        ya = ttnn.permute(ya, (0, 1, 3, 2))
        ya = ttnn.reshape(ya, (ya.shape[0], 4, 16, ya.shape[-1]))
        ya = ttnn.softmax(ya, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ya = ttnn.permute(ya, (0, 1, 3, 2))
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
