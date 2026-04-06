# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov11s.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11s.tt.ttnn_yolov11s_bottleneck import TtnnBottleneck


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
        cfg = self.cv1.conv.conv
        hw = int(cfg.input_height) * int(cfg.input_width)
        k1 = self.k1(device, x1)
        k2 = self.k2(device, k1)
        if x2.is_sharded() and x2.shape[2] > hw:
            x2 = ttnn.sharded_to_interleaved(x2, ttnn.L1_MEMORY_CONFIG)
        if x2.shape[2] > hw:
            x2 = x2[:, :, :hw, :]
        if k2.is_sharded() and k2.shape[2] > hw:
            k2 = ttnn.sharded_to_interleaved(k2, ttnn.L1_MEMORY_CONFIG)
        if k2.shape[2] > hw:
            k2 = k2[:, :, :hw, :]
        if use_shard_concat:
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            k2 = ttnn.to_layout(k2, ttnn.ROW_MAJOR_LAYOUT)
            x = sharded_concat([k2, x2], to_interleaved=False)
        else:
            x = ttnn.concat((k2, x2), dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv3(device, x, output_rm_needed=True)
        deallocate_tensors(x1, x2, k1, k2)
        return x
