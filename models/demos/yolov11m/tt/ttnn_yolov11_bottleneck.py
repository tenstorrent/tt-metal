# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv


class TtnnBottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)

    def __call__(self, device, x, tile_shape=32):
        input = x
        x = self.cv1(device, x)
        x = self.cv2(device, x)
        if x.shape[3] < tile_shape:
            input = ttnn.to_layout(input, layout=ttnn.TILE_LAYOUT)
            x = ttnn.add(input, x, memory_config=x.memory_config())
        else:
            x = ttnn.add(input, x, memory_config=x.memory_config(), use_legacy=False)
        return x
