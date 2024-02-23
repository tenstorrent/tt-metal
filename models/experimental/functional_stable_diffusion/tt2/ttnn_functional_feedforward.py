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
        act = self.geglu(config, hidden_states)
        output = act @ self.parameters.net[2].weight
        output = ttnn.add(output, self.parameters.net[2].bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        return output
