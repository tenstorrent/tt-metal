# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_size": [1],
    "num_heads": [4, 16],
    "sequence_size": [384, 1024],
    "head_size": [64, 128],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}


def run(
    batch_size, num_heads, sequence_size, head_size, input_dtype, input_memory_config, output_memory_config, *, device
) -> Tuple[bool, Optional[str]]:
    input_shape = (batch_size, num_heads, sequence_size, head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = ttnn.transformer._torch_concatenate_heads(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.transformer.concatenate_heads(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
