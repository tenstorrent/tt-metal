# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.mobileNetV3.tt.utils import create_se_conv_config


class ttnn_SqueezeExcitation:
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation=ttnn.relu,
        scale_activation=ttnn.hardsigmoid,
        parameters=None,
    ) -> None:
        super().__init__()
        self.avgpool = ttnn.global_avg_pool2d
        self.input_channels = input_channels
        self.squeeze_channels = squeeze_channels
        self.parameters = parameters
        self.activation = activation
        self.scale_activation = scale_activation
        self.fc1 = None
        self.fc2 = None

    def __call__(self, device, input):
        input = ttnn.to_layout(input, layout=ttnn.ROW_MAJOR_LAYOUT)
        scale = self.avgpool(input)

        if self.fc1 is None and self.fc2 is None:
            self.fc1 = TtConv2d(
                create_se_conv_config((scale.shape[-4], 1, 1, self.input_channels), self.parameters["fc1"]), device
            )
            self.fc2 = TtConv2d(
                create_se_conv_config((scale.shape[-4], 1, 1, self.squeeze_channels), self.parameters["fc2"]), device
            )

        scale = self.fc1(scale)
        scale = self.activation(scale)

        scale = self.fc2(scale)
        scale = self.scale_activation(scale)

        return scale * input
