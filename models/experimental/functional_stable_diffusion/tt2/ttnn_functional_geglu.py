# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class geglu():
    def __init__(self, device, parameters):
        self.device = device
        self.parameters = parameters

    def __call__(self, config, hidden_states):
        output = ttnn.matmul(hidden_states, self.parameters.proj.weight)

        output = ttnn.add(output, self.parameters.proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        hidden_states, gate = ttnn.split(output, split_size=output.shape[-1] // 2, dim=-1)
        del output
        act = ttnn.gelu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)
        del gate
        return ttnn.mul(hidden_states, act)
