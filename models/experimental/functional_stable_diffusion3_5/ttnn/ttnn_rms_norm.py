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
        variance = ttnn.pow(hidden_states, 2)
        variance = ttnn.mean(variance, dim=-1)
        variance = ttnn.add(variance, self.eps)
        variance = ttnn.rsqrt(variance)
        hidden_states = ttnn.multiply(hidden_states, variance)
        if self.weight is not None:
            hidden_states = ttnn.multiply(hidden_states, self.weight)
        return hidden_states
