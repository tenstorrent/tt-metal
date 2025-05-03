# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random, divup


parameters = {
    "batch_sizes": [(1,)],
    "start": [384, 1024],
    "end": [2048, 4096],
    "step": [2, 4, 6],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT],
}


def run(
    batch_sizes,
    start,
    end,
    step,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, start, end, step)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.arange(start, end, step)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    output_tensor = ttnn.arange(
        input_tensor.shape[1],
        input_tensor.shape[2],
        input_tensor.shape[3],
        device=device,
        memory_config=output_memory_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor[-1, -1, -1, :]
    if divup((end - start), step) % 2 != 0:
        output_tensor = output_tensor.view(-1)[:-1]

    return check_with_pcc(torch_output_tensor, output_tensor)
