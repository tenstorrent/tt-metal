# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import models.experimental.bloom.bloom_utils as bloom_utils

mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1)


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

    k1 = torch.full(tuple(x.get_legacy_shape()), 0.5)
    tt_k1 = bloom_utils.torch2tt_tensor(k1, device)

    k2 = torch.full(tuple(x.get_legacy_shape()), 0.044715)
    tt_k2 = bloom_utils.torch2tt_tensor(k2, device)

    k3 = torch.full(tuple(x.get_legacy_shape()), 0.79788456)
    tt_k3 = bloom_utils.torch2tt_tensor(k3, device)

    # 0.5*x
    factor1 = tt_lib.tensor.mul(tt_k1, z, output_mem_config=mem_config)  # exp(z)

    # x*x
    pow2 = tt_lib.tensor.mul(z, z, output_mem_config=mem_config)

    # (x + 0.044715 * torch.pow(x, 3)))
    # torch.pow(x, 3))
    pow3 = tt_lib.tensor.mul(pow2, z, output_mem_config=mem_config)
    factor3 = tt_lib.tensor.mul(tt_k2, pow3, output_mem_config=mem_config)

    # (x + 0.044715 * torch.pow(x, 3)))
    factor3 = tt_lib.tensor.add(factor3, z, output_mem_config=mem_config)

    sumtanh = tt_lib.tensor.mul(tt_k3, factor3, output_mem_config=mem_config)
    tanh = tt_lib.tensor.tanh(sumtanh, output_mem_config=mem_config)

    k4 = torch.full(tuple(x.get_legacy_shape()), 1.0)
    tt_k4 = bloom_utils.torch2tt_tensor(k4, device)

    total = tt_lib.tensor.add(tt_k4, tanh, output_mem_config=mem_config)
    output = tt_lib.tensor.mul(factor1, total, output_mem_config=mem_config)

    return output
