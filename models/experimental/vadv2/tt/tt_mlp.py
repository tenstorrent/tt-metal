# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.tt.common import activation_memory_config


class TtMLP:
    def __init__(self, params, device, in_channels, hidden_unit, verbose=False):
        super(TtMLP, self).__init__()
        self.params = params
        self.device = device

    def __call__(self, x):
        x = ttnn.linear(
            x,
            self.params.linear.weight,
            bias=self.params.linear.bias,
            memory_config=activation_memory_config(self.device, x, self.params.linear.weight.shape[-1]),
        )
        x = ttnn.layer_norm(
            x,
            weight=self.params.norm.weight,
            bias=self.params.norm.bias,
            memory_config=activation_memory_config(self.device, x),
        )
        x = ttnn.relu(x)
        return x
