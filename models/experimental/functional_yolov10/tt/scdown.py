# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_yolov10.tt.common import Conv


class TtnnSCDown:
    def __init__(self, device=None, parameters=None, conv_pt=None):
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
            enable_identity=True,
            use_1d_systolic_array=False,
            auto_shard=True,
        )

    def __call__(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        return x
