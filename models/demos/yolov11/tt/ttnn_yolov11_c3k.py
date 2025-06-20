# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11.tt.ttnn_yolov11_bottleneck import TtnnBottleneck


class TtnnC3K:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)
        self.cv3 = TtnnConv(device, parameter.cv3, conv_pt.cv3, reshard=True)
        self.k1 = TtnnBottleneck(device, parameter.m[0], conv_pt.m[0])
        self.k2 = TtnnBottleneck(device, parameter.m[1], conv_pt.m[1])

    def __call__(self, device, x, use_shard_concat=True):
        x1 = self.cv1(device, x)
        x2 = self.cv2(device, x)
        k1 = self.k1(device, x1)
        k2 = self.k2(device, k1)
        if use_shard_concat:
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            k2 = ttnn.to_layout(k2, ttnn.ROW_MAJOR_LAYOUT)
            x = sharded_concat([k2, x2], to_interleaved=False)
        else:
            x = ttnn.concat((k2, x2), 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv3(device, x)
        deallocate_tensors(x1, x2, k1, k2)
        return x
