# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.common import Conv


class TtnnCIB:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.conv0 = Conv(
            device,
            parameters.cv1[0],
            self.conv_pt.cv1[0],
            use_1d_systolic_array=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
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
            use_1d_systolic_array=False,
        )

        self.conv3 = Conv(
            device,
            parameters.cv1[3],
            self.conv_pt.cv1[3],
            deallocate_activation=True,
        )

        self.conv4 = Conv(
            device,
            parameters.cv1[4],
            self.conv_pt.cv1[4],
            use_1d_systolic_array=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=False,
        )

    def __call__(self, input_tensor):
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        inputs = input_tensor
        conv0_out = self.conv0(input_tensor)
        conv1_out = self.conv1(conv0_out)
        ttnn.deallocate(conv0_out)
        if conv1_out.is_sharded():
            conv1_out = ttnn.sharded_to_interleaved(conv1_out, ttnn.L1_MEMORY_CONFIG)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        output = ttnn.add(inputs, conv4_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        return output
