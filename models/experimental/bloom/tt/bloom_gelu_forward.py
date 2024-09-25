# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import models.experimental.bloom.bloom_utils as bloom_utils

mem_config = ttnn.L1_MEMORY_CONFIG


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def tt_bloom_gelu_forward(x, device):
    z = x

    k1 = torch.full(tuple(x.shape.with_tile_padding()), 0.5)
    tt_k1 = bloom_utils.torch2tt_tensor(k1, device)

    k2 = torch.full(tuple(x.shape.with_tile_padding()), 0.044715)
    tt_k2 = bloom_utils.torch2tt_tensor(k2, device)

    k3 = torch.full(tuple(x.shape.with_tile_padding()), 0.79788456)
    tt_k3 = bloom_utils.torch2tt_tensor(k3, device)

    # 0.5*x
    factor1 = ttnn.mul(tt_k1, z, memory_config=mem_config)  # exp(z)

    # x*x
    pow2 = ttnn.mul(z, z, memory_config=mem_config)

    # (x + 0.044715 * torch.pow(x, 3)))
    # torch.pow(x, 3))
    pow3 = ttnn.mul(pow2, z, memory_config=mem_config)
    factor3 = ttnn.mul(tt_k2, pow3, memory_config=mem_config)

    # (x + 0.044715 * torch.pow(x, 3)))
    factor3 = ttnn.add(factor3, z, memory_config=mem_config)

    sumtanh = ttnn.mul(tt_k3, factor3, memory_config=mem_config)
    tanh = ttnn.tanh(sumtanh, memory_config=mem_config)

    k4 = torch.full(tuple(x.shape.with_tile_padding()), 1.0)
    tt_k4 = bloom_utils.torch2tt_tensor(k4, device)

    total = ttnn.add(tt_k4, tanh, memory_config=mem_config)
    output = ttnn.mul(factor1, total, memory_config=mem_config)

    return output
