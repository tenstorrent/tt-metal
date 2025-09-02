# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11m.tt.common import TtnnConv


class TtnnBottleneck:
    def __init__(self, device, parameter, conv_pt):
        if hasattr(conv_pt.cv1, 'conv') and hasattr(conv_pt.cv1.conv, 'weight'):
            print(f"DEBUG TtnnBottleneck: conv_pt.cv1.conv.weight shape = {conv_pt.cv1.conv.weight.shape}")
        if hasattr(conv_pt.cv2, 'conv') and hasattr(conv_pt.cv2.conv, 'weight'):
            print(f"DEBUG TtnnBottleneck: conv_pt.cv2.conv.weight shape = {conv_pt.cv2.conv.weight.shape}")
        # Check weight shapes (both should be identical as both operate on original input)
        if (hasattr(conv_pt.cv1, 'conv') and hasattr(conv_pt.cv2, 'conv') and 
            hasattr(conv_pt.cv1.conv, 'weight') and hasattr(conv_pt.cv2.conv, 'weight')):
            cv1_shape = conv_pt.cv1.conv.weight.shape
            cv2_shape = conv_pt.cv2.conv.weight.shape
            print(f"DEBUG TtnnBottleneck: cv1 weight: {cv1_shape[1]} → {cv1_shape[0]}")
            print(f"DEBUG TtnnBottleneck: cv2 weight: {cv2_shape[1]} → {cv2_shape[0]}")

        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)

    def __call__(self, device, x, shortcut=False, tile_shape=64):
        input = x
        x = self.cv1(device, x)
        x = self.cv2(device, x)

        return ttnn.add(input, x, memory_config=x.memory_config()) if self.shortcut else x

