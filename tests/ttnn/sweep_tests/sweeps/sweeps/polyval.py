# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1, 2)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "coeff": [(3.6, 23.6, 1.7, 4.6), (9.4, 4.2, 3.3, 9.0)],
}


def torch_polyval(input_tensor, coeff):
    curVal = 0
    for curValIndex in range(len(coeff) - 1):
        curVal = (curVal + coeff[curValIndex]) * input_tensor[0]
    return curVal + coeff[len(coeff) - 1]


def skip(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    coeff,
) -> Tuple[bool, Optional[str]]:
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Skipped as ROW_MAJOR_LAYOUT not supported"
    return False, None


def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    coeff,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_output_tensor = torch_polyval(torch_input_tensor, coeff)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, memory_config=input_memory_config, layout=layout
    )

    output_tensor = ttnn.polyval(input_tensor, coeff, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor).squeeze(0)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
