# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def geglu(config, hidden_states, parameters):
    import torch

    output = ttnn.matmul(hidden_states, parameters.proj.weight)

    output = ttnn.add(output, parameters.proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    output_torch = ttnn.to_torch(output)
    hidden_states_torch, gate_torch = torch.split(output_torch, split_size=output.shape[-1] // 2, dim=-1)
    hidden_states = ttnn.from_torch(hidden_states_torch, device=output.get_device())
    gate = ttnn.from_torch(gate_torch, device=output.get_device())
    del output_torch
    del output

    act = ttnn.gelu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)
    del gate
    del gate_torch

    return ttnn.mul(hidden_states, act)
