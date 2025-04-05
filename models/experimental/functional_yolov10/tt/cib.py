# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_yolov10.tt.common import Conv


class TtnnCIB:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.conv0 = Conv(
            device,
            parameters.cv1[0],
            self.conv_pt.cv1[0],
        )

        self.conv1 = Conv(
            device,
            parameters.cv1[1],
            self.conv_pt.cv1[1],
        )

        self.conv2 = Conv(
            device,
            parameters.cv1[2],
            self.conv_pt.cv1[2],
            auto_shard=True,
        )

        self.conv3 = Conv(
            device,
            parameters.cv1[3],
            self.conv_pt.cv1[3],
        )

        self.conv4 = Conv(
            device,
            parameters.cv1[4],
            self.conv_pt.cv1[4],
        )

    def __call__(self, x):
        input_tensor = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = input_tensor + x
        return x
