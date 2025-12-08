# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.mobileNetV3.tt.utils import create_se_conv_config, post_conv_reshape


class ttnn_SqueezeExcitation:
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation=ttnn.relu,
        scale_activation=ttnn.hardsigmoid,
        parameters=None,
        device=None,
    ) -> None:
        super().__init__()
        self.avgpool = ttnn.global_avg_pool2d
        self.input_channels = input_channels
        self.squeeze_channels = squeeze_channels
        self.parameters = parameters
        self.activation = activation
        self.scale_activation = scale_activation
        self.fc1_config = create_se_conv_config((1, 1, 1, self.input_channels), self.parameters["fc1"])
        self.fc2_config = create_se_conv_config((1, 1, 1, self.squeeze_channels), self.parameters["fc2"])
        self.fc1 = TtConv2d(self.fc1_config, device)
        self.fc2 = TtConv2d(self.fc2_config, device)

    def __call__(self, device, input):
        input = ttnn.to_layout(input, layout=ttnn.ROW_MAJOR_LAYOUT)
        scale = self.avgpool(input)

        scale = self.fc1(scale)
        scale = post_conv_reshape(scale)
        scale = self.activation(scale)

        scale = self.fc2(scale)
        scale = post_conv_reshape(scale)
        scale = self.scale_activation(scale)

        return scale * input
