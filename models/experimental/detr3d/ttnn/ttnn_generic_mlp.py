# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.detr3d.ttnn.common import TtnnConv1D


class TttnnGenericMLP:
    def __init__(
        self,
        # input_dim,
        # hidden_dims,
        # output_dim,
        # norm_fn_name=None,
        # activation="relu",
        # use_conv=False,
        # dropout=None,
        # hidden_use_bias=False,
        # output_use_bias=True,
        # output_use_activation=False,
        # output_use_norm=False,
        # weight_init_name=None,
        module=None,
        parameters=None,
        device=None,
    ):
        self.device = device
        self.parameters = parameters
        self.module = module
        print("module is", module.layers[0])
        print("parameters is", parameters)
        self.tt_layers = []
        for i, layer in enumerate(module.layers):
            # if isinstance(layer, torch.nn.Conv1d):
            #     conv1d_layer = TtnnConv1D(layer, parameters.layers[f"{i}"], device, activation=activation)
            #     self.tt_layers.append(conv1d_layer)
            if isinstance(layer, torch.nn.Conv1d):
                conv1d_layer = TtnnConv1D(layer, parameters.layers[i], device)
                self.tt_layers.append(conv1d_layer)
            elif isinstance(layer, torch.nn.ReLU):
                relu_layer = ttnn.relu
                self.tt_layers.append(relu_layer)

    def __call__(self, x):
        for layer in self.tt_layers:
            x = layer(x)
        return x
