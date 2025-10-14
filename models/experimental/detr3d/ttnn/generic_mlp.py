# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.common import TtnnConv1D


class TtnnGenericMLP(LightweightModule):
    def __init__(
        self,
        module=None,
        parameters=None,
        device=None,
    ):
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.module = module
        self.tt_layers = []
        for i, layer in enumerate(module.layers):
            if isinstance(layer, torch.nn.Conv1d):
                conv1d_layer = TtnnConv1D(layer, parameters.layers[i], device)
                self.tt_layers.append(conv1d_layer)
            elif isinstance(layer, torch.nn.ReLU):
                relu_layer = ttnn.relu
                self.tt_layers.append(relu_layer)

    def forward(self, x):
        for layer in self.tt_layers:
            x = layer(x)
        return x
