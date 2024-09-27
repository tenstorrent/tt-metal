# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1,)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
}


def skip(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
) -> Tuple[bool, Optional[str]]:
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Skipped as ROW_MAJOR_LAYOUT and ttnn.bfloat8_b not supported"
    return False, None


def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16).unsqueeze(0)
    torch_output_tensor = torch.zeros(torch_input_tensor.shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    output_tensor = ttnn.zeros(input_tensor.shape, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor)
