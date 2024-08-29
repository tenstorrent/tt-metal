# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from torch import nn


class TtMLP:
    def __init__(
        self,
        hidden_channels,
        device,
        parameters,
        inplace=None,
        activation_layer=ttnn.relu,
        norm_layer=None,
    ):
        self.params = {} if inplace is None else {"inplace": inplace}
        self.device = device
        self.parameters = parameters
        self.norm_layer = norm_layer
        self.hidden_channels = hidden_channels
        self.activation_layer = activation_layer

    def __call__(self, x):
        for hidden_dim in self.hidden_channels[:-1]:
            x = ttnn.linear(x, self.parameters[0].weight, bias=self.parameters[0].bias)
            if self.norm_layer is not None:
                x = ttnn.layer_norm(x, weight=self.parameters.norm_weight, bias=self.parameters.norm_bias)
            x = self.activation_layer(x)

        x = ttnn.linear(x, self.parameters[3].weight, bias=self.parameters[3].bias)

        return x
