# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, Yolov11Conv2D, deallocate_tensors, sharded_concat_2


class TtnnDetect:
    def __init__(self, device, parameter, conv_pt):
        self.cv2_0_0 = TtnnConv(device, parameter.cv2[0][0], conv_pt.cv2[0][0], is_detect=True)
        self.cv2_0_1 = TtnnConv(device, parameter.cv2[0][1], conv_pt.cv2[0][1], is_detect=True)
        self.cv2_0_2 = Yolov11Conv2D(parameter.cv2[0][2], conv_pt.cv2[0][2], device=device, is_detect=True)

        self.cv2_1_0 = TtnnConv(device, parameter.cv2[1][0], conv_pt.cv2[1][0], is_detect=True)
        self.cv2_1_1 = TtnnConv(device, parameter.cv2[1][1], conv_pt.cv2[1][1], is_detect=True)
        self.cv2_1_2 = Yolov11Conv2D(parameter.cv2[1][2], conv_pt.cv2[1][2], device=device, is_detect=True)

        self.cv2_2_0 = TtnnConv(device, parameter.cv2[2][0], conv_pt.cv2[2][0], is_detect=True)
        self.cv2_2_1 = TtnnConv(device, parameter.cv2[2][1], conv_pt.cv2[2][1], is_detect=True)
        self.cv2_2_2 = Yolov11Conv2D(parameter.cv2[2][2], conv_pt.cv2[2][2], device=device, is_detect=True)

        self.cv3_0_0_0 = TtnnConv(device, parameter.cv3[0][0][0], conv_pt.cv3[0][0][0], is_detect=True)
        self.cv3_0_0_1 = TtnnConv(device, parameter.cv3[0][0][1], conv_pt.cv3[0][0][1], is_detect=True)
        self.cv3_0_1_0 = TtnnConv(device, parameter.cv3[0][1][0], conv_pt.cv3[0][1][0], is_detect=True)
        self.cv3_0_1_1 = TtnnConv(device, parameter.cv3[0][1][1], conv_pt.cv3[0][1][1], is_detect=True)
        self.cv3_0_2_0 = Yolov11Conv2D(parameter.cv3[0][2], conv_pt.cv3[0][2], device=device, is_detect=True)

        self.cv3_1_0_0 = TtnnConv(device, parameter.cv3[1][0][0], conv_pt.cv3[1][0][0], is_detect=True)
        self.cv3_1_0_1 = TtnnConv(device, parameter.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True)
        self.cv3_1_1_0 = TtnnConv(device, parameter.cv3[1][1][0], conv_pt.cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = TtnnConv(device, parameter.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True)
        self.cv3_1_2_0 = Yolov11Conv2D(parameter.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True)

        self.cv3_2_0_0 = TtnnConv(device, parameter.cv3[2][0][0], conv_pt.cv3[2][0][0], is_detect=True)
        self.cv3_2_0_1 = TtnnConv(device, parameter.cv3[2][0][1], conv_pt.cv3[2][0][1], is_detect=True)
        self.cv3_2_1_0 = TtnnConv(device, parameter.cv3[2][1][0], conv_pt.cv3[2][1][0], is_detect=True)
        self.cv3_2_1_1 = TtnnConv(device, parameter.cv3[2][1][1], conv_pt.cv3[2][1][1], is_detect=True)
        self.cv3_2_2_0 = Yolov11Conv2D(parameter.cv3[2][2], conv_pt.cv3[2][2], device=device, is_detect=True)

        self.dfl = Yolov11Conv2D(parameter.dfl.conv, conv_pt.dfl.conv, device=device, is_dfl=True)
        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, device, y1, y2, y3, batch_size=1, tile_size=32):
        bs = batch_size
        x1 = self.cv2_0_0(device, y1, batch_size=bs)
        x1 = self.cv2_0_1(device, x1, batch_size=bs)
        x1 = self.cv2_0_2(x1, batch_size=bs)
        x2 = self.cv2_1_0(device, y2, batch_size=bs)
        x2 = self.cv2_1_1(device, x2, batch_size=bs)
        x2 = self.cv2_1_2(x2, batch_size=bs)

        x3 = self.cv2_2_0(device, y3, batch_size=bs)
        x3 = self.cv2_2_1(device, x3, batch_size=bs)
        x3 = self.cv2_2_2(x3, batch_size=bs)

        x4 = self.cv3_0_0_0(device, y1, batch_size=bs)
        x4 = self.cv3_0_0_1(device, x4, batch_size=bs)
        x4 = self.cv3_0_1_0(device, x4, batch_size=bs)
        x4 = self.cv3_0_1_1(device, x4, batch_size=bs)
        x4 = self.cv3_0_2_0(x4, batch_size=bs)

        x5 = self.cv3_1_0_0(device, y2, batch_size=bs)
        x5 = self.cv3_1_0_1(device, x5, batch_size=bs)
        x5 = self.cv3_1_1_0(device, x5, batch_size=bs)
        x5 = self.cv3_1_1_1(device, x5, batch_size=bs)
        x5 = self.cv3_1_2_0(x5, batch_size=bs)

        x6 = self.cv3_2_0_0(device, y3, batch_size=bs)
        x6 = self.cv3_2_0_1(device, x6, batch_size=bs)
        x6 = self.cv3_2_1_0(device, x6, batch_size=bs)
        x6 = self.cv3_2_1_1(device, x6, batch_size=bs)
        x6 = self.cv3_2_2_0(x6, batch_size=bs)

        y1 = sharded_concat_2(x1, x4)
        y2 = sharded_concat_2(x2, x5)
        y3 = sharded_concat_2(x3, x6)

        y1 = ttnn.sharded_to_interleaved(y1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.sharded_to_interleaved(y2, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3 = ttnn.sharded_to_interleaved(y3, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Keep batch in dim 0: reshape each scale [1,1,N*HiWi,144] -> [N, HiWi, 144]
        # and concat over the anchor axis (dim=1) -> [N, 8400, 144]. The original
        # concat(dim=2)+squeeze(dim=0) is the N==1 special case of this.
        y1 = ttnn.reshape(y1, (bs, y1.shape[2] // bs, y1.shape[3]))
        y2 = ttnn.reshape(y2, (bs, y2.shape[2] // bs, y2.shape[3]))
        y3 = ttnn.reshape(y3, (bs, y3.shape[2] // bs, y3.shape[3]))
        y = ttnn.concat((y1, y2, y3), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT)
        ya, yb = y[:, :, :64], y[:, :, 64:144]
        deallocate_tensors(y1, y2, y3, x1, x2, x3, x4, x5, x6, y)
        ya = ttnn.reshape(ya, (ya.shape[0], y.shape[1], 4, 16))
        ya = ttnn.softmax_in_place(ya, dim=-1, numeric_stable=False)
        ya = ttnn.permute(ya, (0, 2, 1, 3))
        c = self.dfl(ya)
        ttnn.deallocate(ya)
        c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.permute(c, (0, 3, 1, 2))
        # DFL output is collapsed image-major as [1,1,1,N*4*8400] with per-image
        # order (coord in 4, anchor in 8400); split batch back into dim 0 so the
        # [1,2,8400] anchors broadcast per image. (N==1 -> [1,4,8400] as before.)
        c = ttnn.reshape(c, (bs, 4, int(c.shape[3] / (4 * bs))))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]
        anchor, strides = self.anchors, self.strides
        anchor = ttnn.to_memory_config(anchor, memory_config=ttnn.L1_MEMORY_CONFIG)
        strides = ttnn.to_memory_config(strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        c1 = anchor - c1
        c2 = anchor + c2
        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)
        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        z = ttnn.multiply(z, strides)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)
        deallocate_tensors(c, z1, z2, c1, c2, anchor, strides)
        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        deallocate_tensors(yb, z)
        return out
