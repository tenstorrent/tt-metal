# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov10.tt.common import Conv


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
            auto_shard=True,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            auto_shard=True,
        )

    def __call__(self, x):
        cv1 = self.cv1(x)
        cv2 = self.cv2(cv1)
        ttnn.deallocate(cv1)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv2
