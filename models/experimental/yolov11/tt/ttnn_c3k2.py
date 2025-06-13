# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov11.tt.common import Conv, deallocate_tensors, sharded_concat
from models.experimental.yolov11.tt.ttnn_c3k import C3K
from models.experimental.yolov11.tt.ttnn_bottleneck import Bottleneck


class C3k2:
    def __init__(self, device, parameter, conv_pt, is_bk_enabled=False):
        self.is_bk_enabled = is_bk_enabled
        self.parameter = parameter

        if is_bk_enabled:
            self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
            self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
            self.k = Bottleneck(device, parameter[0], conv_pt.m[0])
        else:
            self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
            self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
            self.c3k = C3K(device, parameter[0], conv_pt.m[0])

    def __call__(self, device, x):
        x = self.cv1(device, x)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        y1 = x[:, :, :, : x.shape[-1] // 2]
        y2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        if self.is_bk_enabled:
            y2 = ttnn.to_layout(y2, layout=ttnn.TILE_LAYOUT)
            y3 = self.k(device, y2)
        else:
            y3 = self.c3k(device, y2)

        if y2.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            y2 = ttnn.to_layout(y2, ttnn.ROW_MAJOR_LAYOUT)
        if y3.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            y3 = ttnn.to_layout(y3, ttnn.ROW_MAJOR_LAYOUT)
        use_shard_concat = True
        if use_shard_concat:
            x = sharded_concat([y1, y2, y3])
        else:
            x = ttnn.concat((y1, y2, y3), 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(device, x)

        deallocate_tensors(y1, y2, y3)
        return x
