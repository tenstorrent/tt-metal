# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "ttnn_function,torch_function": [
        (ttnn.exp, torch.exp),
        (ttnn.tanh, torch.tanh),
        (ttnn.gelu, torch.nn.functional.gelu),
        (ttnn.rsqrt, torch.rsqrt),
        (ttnn.relu, torch.relu),
    ],
    "batch_sizes": [(1,)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}


def skip(**_):
    return False


def run(
    ttnn_function,
    torch_function,
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    *,
    device,
):
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, dtype=input_dtype, memory_config=input_memory_config
    )

    output_tensor = ttnn_function(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
