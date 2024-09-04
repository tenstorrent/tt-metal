# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1, 1)],
    "height": [32, 64, 96, 128, 160, 192, 224, 256],
    "width": [64],
    "token_index": [0],
    "input_dtype": [ttnn.bfloat8_b, ttnn.bfloat16],
    "input_memory_config": [ttnn.L1_MEMORY_CONFIG],
    "output_memory_config": [ttnn.L1_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT],
}


def run(
    batch_sizes,
    height,
    width,
    token_index,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    cache_size = 2048
    sin_cos_shape = (1, 1, cache_size, 64)

    torch_cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    torch_sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    input_shape = (*batch_sizes, height, width)

    low = -1
    high = 1

    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.experimental.rotary_embedding)
    torch_output_tensor = golden_function(torch_input_tensor, torch_cos_cached, torch_sin_cached, token_index)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, memory_config=input_memory_config, layout=layout
    )
    cos_cached = ttnn.from_torch(
        torch_cos_cached, dtype=input_dtype, device=device, memory_config=input_memory_config, layout=layout
    )
    sin_cached = ttnn.from_torch(
        torch_sin_cached, dtype=input_dtype, device=device, memory_config=input_memory_config, layout=layout
    )

    output_tensor = ttnn.experimental.rotary_embedding(
        input_tensor, cos_cached, sin_cached, token_index, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
