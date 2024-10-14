# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_mobilenetv3.ttnn.ttnn_invertedResidual import (
    ttnn_InvertedResidual,
    InvertedResidualConfig,
    Conv2dNormActivation,
)
from typing import Any, Callable, List, Optional, Sequence


class ttnn_MobileNetV3:
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block=None,
        parameters=None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = ttnn_InvertedResidual

        layers = []
        index = 0
        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                kernel_size=3, stride=2, activation_layer=ttnn.hardswish, parameters=parameters["features"][index]
            )
        )

        index += 1

        # building inverted residual blocks
        for i, cnf in enumerate(inverted_residual_setting):
            layers.append(block(cnf, parameters=parameters["features"][index].block))
            index += 1

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                kernel_size=1,
                activation_layer=ttnn.hardswish,
                parameters=parameters["features"][index],
            )
        )

        self.features = layers
        self.avgpool = ttnn.global_avg_pool2d
        self.classifier = [
            ttnn.linear,
            ttnn.hardswish,
            ttnn.linear,
        ]
        self.parameters = parameters

    def __call__(self, device, x):
        for i, layer in enumerate(self.features):
            x = layer(device, x)

        x = self.avgpool(x)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], -1))
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

        self.parameters["classifier"][0].weight = ttnn.to_device(self.parameters["classifier"][0].weight, device=device)
        self.parameters["classifier"][3].weight = ttnn.to_device(self.parameters["classifier"][3].weight, device=device)
        self.parameters["classifier"][0].bias = ttnn.to_device(self.parameters["classifier"][0].bias, device=device)
        self.parameters["classifier"][3].bias = ttnn.to_device(self.parameters["classifier"][3].bias, device=device)

        x = self.classifier[0](x, self.parameters["classifier"][0].weight, bias=self.parameters["classifier"][0].bias)
        x = self.classifier[1](x)
        x = self.classifier[2](x, self.parameters["classifier"][3].weight, bias=self.parameters["classifier"][3].bias)

        return x
