# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11m.tt.common import TtnnConv


class TtnnBottleneck:
    def __init__(self, device, parameter, conv_pt):
        print(f"DEBUG TtnnBottleneck: parameter object = {type(parameter)}")
        print(f"DEBUG TtnnBottleneck: parameter.cv1 = {type(parameter.cv1) if hasattr(parameter, 'cv1') else 'NO cv1'}")
        print(f"DEBUG TtnnBottleneck: parameter.cv2 = {type(parameter.cv2) if hasattr(parameter, 'cv2') else 'NO cv2'}")
        print(f"DEBUG TtnnBottleneck: conv_pt object = {type(conv_pt)}")
        print(f"DEBUG TtnnBottleneck: conv_pt.cv1 channels = {conv_pt.cv1.conv.in_channels} -> {conv_pt.cv1.conv.out_channels}")
        print(f"DEBUG TtnnBottleneck: conv_pt.cv2 channels = {conv_pt.cv2.conv.in_channels} -> {conv_pt.cv2.conv.out_channels}")
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)

    def __call__(self, device, x, tile_shape=64):
        print(f"DEBUG Bottleneck: Input shape = {x.shape}")
        input = x
        x = self.cv1(device, x)
        print(f"DEBUG Bottleneck: After cv1 shape = {x.shape}")
        x = self.cv2(device, x)
        print(f"DEBUG Bottleneck: After cv2 shape = {x.shape}")
        if x.shape[3] < tile_shape:
            input = ttnn.to_layout(input, layout=ttnn.TILE_LAYOUT)
            x = ttnn.add(input, x, memory_config=x.memory_config())
        else:
            x = ttnn.add(input, x, memory_config=x.memory_config(), use_legacy=False)
        return x
