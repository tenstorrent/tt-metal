# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_geglu import geglu


def feedforward(config, hidden_states, parameters, device):
    act = geglu(config, hidden_states, parameters.net[0], device)
    output = act @ parameters.net[2].weight
    output = ttnn.add(output, parameters.net[2].bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    return output
