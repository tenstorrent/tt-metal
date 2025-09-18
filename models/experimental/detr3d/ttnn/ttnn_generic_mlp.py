# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.detr3d.ttnn.common import TtnnConv1D
import torch


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TttnnGenericMLP:
    def __init__(self, module, parameters, device):
        self.device = device
        self.parameters = parameters
        self.module = module
        print("module is", module.layers[0])
        print("parameters is", parameters)
        self.tt_layers = []
        for i, layer in enumerate(module.layers):
            # dropout
            if isinstance(layer, torch.nn.Dropout):
                print(f"Skipping Dropout at layers[{i}]")
            # bn1d  - torch
            elif isinstance(layer, torch.nn.BatchNorm1d):
                print(f"Adding BatchNorm1d from layers[{i}]")
                self.tt_layers.append(layer)
            elif isinstance(layer, torch.nn.Conv1d):
                conv1d_layer = TtnnConv1D(module.layers[i], parameters.layers[i], device)
                self.tt_layers.append(conv1d_layer)
            elif isinstance(layer, torch.nn.ReLU):
                relu_layer = ttnn.relu
                self.tt_layers.append(relu_layer)

        print(self.tt_layers)

    def __call__(self, x):
        for i, layer in enumerate(self.tt_layers):
            if isinstance(layer, torch.nn.BatchNorm1d):
                if x.is_sharded():
                    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
                x = ttnn.to_torch(x).squeeze(dim=0).permute(0, 2, 1)
                x = layer(x).unsqueeze(dim=0).permute(0, 1, 3, 2)  # torch bn1d
                x = ttnn.from_torch(
                    x,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                x = layer(x)
                print(f"ttnn layer no {i}out is", x.shape)
        return x
