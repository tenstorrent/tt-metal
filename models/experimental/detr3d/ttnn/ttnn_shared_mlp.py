# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.detr3d.ttnn.common import TtnnConv2D


class TtnnSharedMLP:
    def __init__(self, module, parameters, device):
        self.device = device
        self.parameters = parameters
        self.conv1 = TtnnConv2D(
            module.layer0.conv,
            parameters.layer0.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=False,
        )
        self.conv2 = TtnnConv2D(
            module.layer1.conv,
            parameters.layer1.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=False,
        )
        self.conv3 = TtnnConv2D(
            module.layer2.conv,
            parameters.layer2.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=False,
        )

    def __call__(self, features):
        # print(f"{features.shape=}")
        conv1 = self.conv1(features)
        # print(f"{conv1.shape=}")
        conv2 = self.conv2(conv1)
        # print(f"{conv2.shape=}")
        conv3 = self.conv3(conv2)
        # print(f"{conv3.shape=}")
        # print(f"{_=}")
        return conv3
