# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.common import Conv


class TtnnBottleNeck:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            deallocate_activation=True,
        )

    def __call__(self, input_tensor):
        cv1 = self.cv1(input_tensor)
        cv2 = self.cv2(cv1)
        if input_tensor.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        return ttnn.add(input_tensor, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv2
