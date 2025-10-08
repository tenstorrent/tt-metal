# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.mobileNetV3.tt.utils import Conv


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
        self.fc1 = Conv(1, 0, parameters=parameters["fc1"], activation=None)
        self.fc2 = Conv(1, 0, parameters=parameters["fc2"], activation=None)
        self.activation = activation
        self.scale_activation = scale_activation

    def __call__(self, device, input):
        input = ttnn.to_layout(input, layout=ttnn.ROW_MAJOR_LAYOUT)
        scale = self.avgpool(input)
        scale = self.fc1(device, scale)
        scale = self.activation(scale)
        scale = self.fc2(device, scale)
        scale = self.scale_activation(scale)
        return scale * input
