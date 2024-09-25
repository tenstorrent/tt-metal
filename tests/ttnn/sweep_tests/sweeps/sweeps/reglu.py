# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1,)],
    "height": [32, 384, 1024],
    "width": [64, 1024, 4096],
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
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return True, "This combination is not supported"
    return False, None


def torch_reglu(input_tensor, *args, **kwargs):
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * F.relu(tensB)


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
    input_shape = (1, *batch_sizes, height, width)

    low = -100.0
    high = 100.0

    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.float32)
    torch_output_tensor = torch_reglu(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    output_tensor = ttnn.reglu(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor)
