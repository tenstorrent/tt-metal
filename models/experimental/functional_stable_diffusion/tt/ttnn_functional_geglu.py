# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def geglu(config, hidden_states, parameters):
    # output = ttnn.linear(hidden_states, parameters.proj.weight, bias = parameters.proj.bias)
    output = ttnn.matmul(hidden_states, parameters.proj.weight)
    output = ttnn.add(output, parameters.proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    hidden_states, gate = ttnn.split(output, split_size=output.shape[-1] // 2, dim=-1)
    del output
    act = ttnn.gelu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)
    del gate
    return ttnn.mul(hidden_states, act, memory_config=ttnn.L1_MEMORY_CONFIG)
