# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch


class ttnn_RMSNorm:
    def __init__(self, dim, eps, elementwise_affine, parameters):
        self.eps = eps
        if elementwise_affine:
            self.weight = parameters.weight
        else:
            self.weight = None

    def __call__(self, hidden_states, device):
        print("hidden_states config before pow", hidden_states.memory_config())
        variance = ttnn.pow(hidden_states, 2)
        print("hidden_states config after pow", hidden_states.memory_config())
        variance = ttnn.mean(variance, dim=-1)
        print("variance config after mean", variance.memory_config())
        variance = ttnn.add(variance, self.eps)
        print("variance config after add", variance.memory_config())
        variance = ttnn.rsqrt(variance)
        print("variance config after rsqrt", variance.memory_config())
        hidden_states = ttnn.multiply(hidden_states, variance)
        print("hidden_states config after mul", hidden_states.memory_config())
        if self.weight is not None:
            hidden_states = ttnn.multiply(hidden_states, self.weight)
            print("hidden_states config after mean", hidden_states.memory_config())
        return hidden_states
