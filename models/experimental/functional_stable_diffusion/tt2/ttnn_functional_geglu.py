# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def split_linear_params(params):
    dim = -1
    device = params.proj.weight.device()
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    weight = ttnn_to_torch(params.proj.weight)
    bias = ttnn_to_torch(params.proj.bias)

    proj_weight, gate_weight = torch.split(weight, weight.shape[dim] // 2, dim=dim)
    proj_bias, gate_bias = torch.split(bias, bias.shape[dim] // 2, dim=dim)

    proj_weight = ttnn.from_torch(proj_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    proj_weight = ttnn.to_device(proj_weight, device, memory_config=memory_config)

    gate_weight = ttnn.from_torch(gate_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    gate_weight = ttnn.to_device(gate_weight, device, memory_config=memory_config)

    proj_bias = ttnn.from_torch(proj_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    proj_bias = ttnn.to_device(proj_bias, device, memory_config=memory_config)

    gate_bias = ttnn.from_torch(gate_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    gate_bias = ttnn.to_device(gate_bias, device, memory_config=memory_config)

    params.proj.proj_weight = proj_weight
    params.proj.gate_weight = gate_weight
    params.proj.proj_bias = proj_bias
    params.proj.gate_bias = gate_bias

    del params.proj.weight
    del params.proj.bias
    return params


class geglu:
    def __init__(self, device, parameters):
        self.device = device
        parameters = split_linear_params(parameters)
        self.parameters = parameters

    def __call__(self, config, hidden_states):
        proj = ttnn.linear(
            hidden_states,
            self.parameters.proj.proj_weight,
            bias=self.parameters.proj.proj_bias,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        gate = ttnn.linear(
            hidden_states,
            self.parameters.proj.gate_weight,
            bias=self.parameters.proj.gate_bias,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            activation="gelu",
        )
        return ttnn.mul(proj, gate)
