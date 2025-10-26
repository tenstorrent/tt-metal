# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.common import Conv


class TtnnSCDown:
    def __init__(self, device=None, parameters=None, conv_pt=None):
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
            enable_identity=True,
            use_1d_systolic_array=False,
            deallocate_activation=True,
        )

    def __call__(self, input_tensor):
        cv1 = self.cv1(input_tensor)
        cv1 = ttnn.sharded_to_interleaved(
            cv1, ttnn.L1_MEMORY_CONFIG
        )  # needed since cv2 uses block_sharding and input is in height sharding
        output = self.cv2(cv1)
        return output
