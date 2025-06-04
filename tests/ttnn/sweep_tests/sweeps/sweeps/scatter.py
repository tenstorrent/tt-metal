# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from typing import Optional, Tuple

from models.utility_functions import torch_random
from utils_for_testing import assert_equal

parameters = {
    "batch_sizes": [(1,)],
    "height": [1, 1],
    "width": [1, 151936],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT],
}


def run(
    batch_sizes, height, width, input_dtype, input_memory_config, output_memory_config, layout, *, device
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)
    scatter_dim = -1
    low = 1e-5
    high = 1e2
    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.bfloat16)
    torch_index_tensor = torch_random(input_shape, 0, input_shape[scatter_dim], dtype=torch.int64)
    torch_source_tensor = torch_random(input_shape, low, high, dtype=torch.bfloat16)
    torch_output_tensor = torch.scatter(
        torch_input_tensor, scatter_dim, index=torch_index_tensor, src=torch_source_tensor
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )
    index_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )
    source_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    output_tensor = ttnn.experimental.scatter_(
        input_tensor, scatter_dim, index_tensor, source_tensor, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)

    return assert_equal(torch_output_tensor, output_tensor)
