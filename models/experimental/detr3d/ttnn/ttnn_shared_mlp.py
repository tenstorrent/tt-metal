# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.detr3d.ttnn.common import TtnnConv2D


class TtnnSharedMLP:
    def __init__(self, module, parameters, device):
        self.device = device
        self.parameters = parameters
        self.conv1 = TtnnConv2D(module.layer0.conv, parameters.layer0.conv, device, activation="relu")
        self.conv2 = TtnnConv2D(module.layer1.conv, parameters.layer1.conv, device, activation="relu")
        self.conv3 = TtnnConv2D(module.layer2.conv, parameters.layer2.conv, device, activation="relu")

    def __call__(self, features):
        conv1 = self.conv1(features)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return conv3
