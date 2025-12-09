# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.mobileNetV3.tt.utils import _create_conv_config_from_params


class ttnn_SqueezeExcitation:
    """
    Squeeze-Excitation block using TT-CNN Builder API.
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        batch_size: int = 1,
        activation=ttnn.relu,
        scale_activation=ttnn.hardsigmoid,
        parameters=None,
        device=None,
    ) -> None:
        super().__init__()
        self.avgpool = ttnn.global_avg_pool2d
        self.activation = activation
        self.scale_activation = scale_activation

        self.fc1_config = _create_conv_config_from_params(
            input_height=1,
            input_width=1,
            in_channels=input_channels,
            out_channels=squeeze_channels,
            batch_size=batch_size,
            parameters=parameters["fc1"],
        )
        self.fc2_config = _create_conv_config_from_params(
            input_height=1,
            input_width=1,
            in_channels=squeeze_channels,
            out_channels=input_channels,
            batch_size=batch_size,
            parameters=parameters["fc2"],
        )

        self.fc1 = TtConv2d(self.fc1_config, device)
        self.fc2 = TtConv2d(self.fc2_config, device)

    def __call__(self, device, input):
        input = ttnn.to_layout(input, layout=ttnn.ROW_MAJOR_LAYOUT)
        scale = self.avgpool(input)

        # if self.fc1 is None:
        #     self.fc1 = TtConv2d(self.fc1_config, device)
        #     self.fc2 = TtConv2d(self.fc2_config, device)

        [scale, [_out_height, _out_width]] = self.fc1(scale, return_output_dim=True)
        # scale = post_conv_reshape(scale)
        scale = self.activation(scale)

        [scale, [_out_height, _out_width]] = self.fc2(scale, return_output_dim=True)
        # scale = post_conv_reshape(scale)
        scale = self.scale_activation(scale)

        # return scale * input
        return scale * input
