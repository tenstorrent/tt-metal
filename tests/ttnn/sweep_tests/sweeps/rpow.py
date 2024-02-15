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
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT],
    "scalar": [1, 2, 5, 10],
}


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def torch_rpow(x, scalar):
    result = torch.pow(scalar, x)
    return result


def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    scalar,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_output_tensor = torch_rpow(torch_input_tensor, scalar)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    output_tensor = ttnn.rpow(input_tensor, scalar, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
