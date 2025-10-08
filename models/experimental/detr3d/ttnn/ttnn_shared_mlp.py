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
            is_dealloc_act=False,
            return_dims=True,
        )
        self.conv2 = TtnnConv2D(
            module.layer1.conv,
            parameters.layer1.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=True,
        )
        self.conv3 = TtnnConv2D(
            module.layer2.conv,
            parameters.layer2.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=True,
        )

    def __call__(self, features):
        shape = features.shape
        print(f"{features.shape=}")
        print(f"{shape=}")
        conv1, shape = self.conv1(features, shape)
        print(f"{conv1.shape=}")
        print(f"{shape=}")
        conv2, shape = self.conv2(conv1, shape)
        print(f"{conv2.shape=}")
        print(f"{shape=}")
        conv3, shape = self.conv3(conv2, shape)
        print(f"{conv3.shape=}")
        print(f"{shape=}")
        conv3 = ttnn.reshape(conv3, shape)
        return conv3
