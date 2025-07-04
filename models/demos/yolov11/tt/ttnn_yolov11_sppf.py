# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors, sharded_concat


class TtnnSPPF:
    def __init__(self, device, parameter, conv_pt):
        self.parameter = parameter
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=True)

    def __call__(self, device, x, use_sharded_concat=True):
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
        if use_sharded_concat:
            y = sharded_concat([x1, m1, m2, m3], to_interleaved=False)
        else:
            y = ttnn.concat([x1, m1, m2, m3], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(device, y)
        deallocate_tensors(x1, m1, m2, m3)
        return x
