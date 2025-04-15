# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.yolov10.tt.common import Conv


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

    def __call__(self, input_tensor):
        inputs = input_tensor
        conv0_out = self.conv0(input_tensor)
        conv1_out = self.conv1(conv0_out)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        output = inputs + conv4_out
        return output
