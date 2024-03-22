# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_geglu import geglu


class feedforward:
    def __init__(self, device, parameters):
        self.device = device
        self.parameters = parameters
        self.geglu = geglu(device, parameters.net[0])

    def __call__(self, config, hidden_states):
        hidden_states = self.geglu(config, hidden_states)
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters.net[2].weight,
            bias=self.parameters.net[2].bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        return hidden_states
