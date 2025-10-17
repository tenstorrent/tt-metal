# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.common import TtnnConv1D


class TtnnGenericMLP(LightweightModule):
    def __init__(
        self,
        module,
        parameters,
        device,
    ):
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.tt_layers = list()

        for layer_num, layer in enumerate(module.layers):
            if isinstance(layer, torch.nn.Conv1d):
                activation = None
                # Checking the next 2 consecutive layers for activation in case batchnorm is next layer
                for index in range(1, 3):
                    if (layer_num + index) < len(module.layers):
                        successor_layer = module.layers[layer_num + index]
                        if isinstance(successor_layer, torch.nn.ReLU):
                            activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
                            break
                    else:
                        break
                self.tt_layers.append(
                    TtnnConv1D(
                        layer,
                        parameters.layers[layer_num],
                        device,
                        activation=activation,
                        return_dims=True,
                        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    )
                )

    def forward(self, x):
        shape = x.shape
        for layer in self.tt_layers:
            x, shape = layer(x, shape)
        x = ttnn.reshape(x, shape)
        return x
