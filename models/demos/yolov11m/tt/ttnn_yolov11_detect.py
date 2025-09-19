# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11m.tt.common import TtnnConv, Yolov11Conv2D, deallocate_tensors


class TtnnDetect:
    def __init__(self, device, parameter, conv_pt):
        self.cv2_0_0 = TtnnConv(device, parameter.cv2[0][0], conv_pt.cv2[0][0], is_detect=True)
        self.cv2_0_1 = TtnnConv(device, parameter.cv2[0][1], conv_pt.cv2[0][1], is_detect=True, deallocate_activation=True)
        self.cv2_0_2 = Yolov11Conv2D(parameter.cv2[0][2], conv_pt.cv2[0][2], device=device, is_detect=True, deallocate_activation=True)

        self.cv2_1_0 = TtnnConv(device, parameter.cv2[1][0], conv_pt.cv2[1][0], is_detect=True)
        self.cv2_1_1 = TtnnConv(device, parameter.cv2[1][1], conv_pt.cv2[1][1], is_detect=True, deallocate_activation=True)
        self.cv2_1_2 = Yolov11Conv2D(parameter.cv2[1][2], conv_pt.cv2[1][2], device=device, is_detect=True, deallocate_activation=True)

        self.cv2_2_0 = TtnnConv(device, parameter.cv2[2][0], conv_pt.cv2[2][0], is_detect=True)
        self.cv2_2_1 = TtnnConv(device, parameter.cv2[2][1], conv_pt.cv2[2][1], is_detect=True, deallocate_activation=True)
        self.cv2_2_2 = Yolov11Conv2D(parameter.cv2[2][2], conv_pt.cv2[2][2], device=device, is_detect=True, deallocate_activation=True)

        self.cv3_0_0_0 = TtnnConv(device, parameter.cv3[0][0][0], conv_pt.cv3[0][0][0], is_detect=True)
        self.cv3_0_0_1 = TtnnConv(device, parameter.cv3[0][0][1], conv_pt.cv3[0][0][1], is_detect=True, deallocate_activation=True)
        self.cv3_0_1_0 = TtnnConv(device, parameter.cv3[0][1][0], conv_pt.cv3[0][1][0], is_detect=True, deallocate_activation=True)
        self.cv3_0_1_1 = TtnnConv(device, parameter.cv3[0][1][1], conv_pt.cv3[0][1][1], is_detect=True, deallocate_activation=True)
        self.cv3_0_2_0 = Yolov11Conv2D(parameter.cv3[0][2], conv_pt.cv3[0][2], device=device, is_detect=True, deallocate_activation=True)

        self.cv3_1_0_0 = TtnnConv(device, parameter.cv3[1][0][0], conv_pt.cv3[1][0][0], is_detect=True)
        self.cv3_1_0_1 = TtnnConv(device, parameter.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True, deallocate_activation=True)
        self.cv3_1_1_0 = TtnnConv(device, parameter.cv3[1][1][0], conv_pt.cv3[1][1][0], is_detect=True, deallocate_activation=True)
        self.cv3_1_1_1 = TtnnConv(device, parameter.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True, deallocate_activation=True)
        self.cv3_1_2_0 = Yolov11Conv2D(parameter.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True, deallocate_activation=True)

        self.cv3_2_0_0 = TtnnConv(device, parameter.cv3[2][0][0], conv_pt.cv3[2][0][0], is_detect=True, deallocate_activation=True)
        self.cv3_2_0_1 = TtnnConv(device, parameter.cv3[2][0][1], conv_pt.cv3[2][0][1], is_detect=True, deallocate_activation=True)
        self.cv3_2_1_0 = TtnnConv(device, parameter.cv3[2][1][0], conv_pt.cv3[2][1][0], is_detect=True, deallocate_activation=True)
        self.cv3_2_1_1 = TtnnConv(device, parameter.cv3[2][1][1], conv_pt.cv3[2][1][1], is_detect=True, deallocate_activation=True)
        self.cv3_2_2_0 = Yolov11Conv2D(parameter.cv3[2][2], conv_pt.cv3[2][2], device=device, is_detect=True, deallocate_activation=True)

        self.dfl = Yolov11Conv2D(parameter.dfl.conv, conv_pt.dfl.conv, device=device, is_dfl=True)
        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, device, y1, y2, y3):
        # Process cv2 branch first
        x1 = self.cv2_0_0(device, y1)
        x1 = self.cv2_0_1(device, x1)
        x1 = self.cv2_0_2(x1)
        x2 = self.cv2_1_0(device, y2)
        x2 = self.cv2_1_1(device, x2)
        x2 = self.cv2_1_2(x2)

        x3 = self.cv2_2_0(device, y3)
        x3 = self.cv2_2_1(device, x3)
        x3 = self.cv2_2_2(x3)

        # Process cv3 branch and deallocate input tensors when done
        x4 = self.cv3_0_0_0(device, y1)
        x4 = self.cv3_0_0_1(device, x4)
        x4 = self.cv3_0_1_0(device, x4)
        x4 = self.cv3_0_1_1(device, x4)
        x4 = self.cv3_0_2_0(x4)
        # y1 no longer needed after x4 is computed
        ttnn.deallocate(y1)

        x1_sharded = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x1)
        x4_sharded = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x4)
        # Concatenate and deallocate intermediate tensors immediately after use
        y1_result = ttnn.concat((x1_sharded, x4_sharded), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        deallocate_tensors(x1_sharded, x4_sharded)  # Free x1, x4 immediately after concat

        x5 = self.cv3_1_0_0(device, y2)
        x5 = self.cv3_1_0_1(device, x5)
        x5 = self.cv3_1_1_0(device, x5)
        x5 = self.cv3_1_1_1(device, x5)
        x5 = self.cv3_1_2_0(x5)
        # y2 no longer needed after x5 is computed
        ttnn.deallocate(y2)

        x2_sharded = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x2)
        x5_sharded = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x5)
        y2_result = ttnn.concat((x2_sharded, x5_sharded), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x2_sharded)
        ttnn.deallocate(x5_sharded)

        x6 = self.cv3_2_0_0(device, y3)
        x6 = self.cv3_2_0_1(device, x6)
        x6 = self.cv3_2_1_0(device, x6)
        x6 = self.cv3_2_1_1(device, x6)
        x6 = self.cv3_2_2_0(x6)
        # y3 no longer needed after x6 is computed
        ttnn.deallocate(y3)

        x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)
        

        
 
        y3_result = ttnn.concat((x3, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        deallocate_tensors(x3, x6)  # Free x3, x6 immediately after concat
        
        y = ttnn.concat((y1_result, y2_result, y3_result), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        deallocate_tensors(y1_result, y2_result, y3_result)  # Free intermediate concat results
        
        y = ttnn.squeeze(y, dim=0)
        y_shape_1 = y.shape[1]  # Store shape before deallocating
        ya, yb = y[:, :, :64], y[:, :, 64:144]
        deallocate_tensors(y)  # Free y after splitting
        ya = ttnn.reallocate(ya)
        yb = ttnn.reallocate(yb)
        ya = ttnn.reshape(ya, (ya.shape[0], y_shape_1, 4, 16))
        ya = ttnn.softmax(ya, dim=-1)
        ya = ttnn.permute(ya, (0, 2, 1, 3))
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
        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)
        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        z = ttnn.multiply(z, strides)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)
        deallocate_tensors(c, z1, z2, c1, c2, anchor, strides)
        z = ttnn.reallocate(z)
        yb = ttnn.reallocate(yb)
        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        deallocate_tensors(yb, z)
        return out
