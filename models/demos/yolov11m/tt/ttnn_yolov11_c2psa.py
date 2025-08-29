# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors
from models.demos.yolov11.tt.ttnn_yolov11_psa import TtnnPSABlock


class TtnnC2PSA:
    def __init__(self, device, parameter, conv_pt):
        self.out_channel_0 = parameter.cv1.conv.out_channels
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)
        self.psablock = TtnnPSABlock(device, parameter.m[0], conv_pt.m[0])

    def __call__(self, device, x):
        x = self.cv1(device, x)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        a, b = x[:, :, :, : int(self.out_channel_0 / 2)], x[:, :, :, int(self.out_channel_0 / 2) :]
        x = self.psablock(device, b)
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.concat((a, x), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(device, x)
        deallocate_tensors(a, b)

        return x
