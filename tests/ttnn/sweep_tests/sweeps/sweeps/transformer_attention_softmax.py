# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_size": [1],
    "num_heads": [1],
    "sequence_size": [384, 1024],
    "target_sequence_size": [384, 4096],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}


def run(
    batch_size,
    num_heads,
    sequence_size,
    target_sequence_size,
    input_dtype,
    input_memory_config,
    output_memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (batch_size, num_heads, sequence_size, target_sequence_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = ttnn.transformer._torch_attention_softmax(
        torch_input_tensor,
        head_size=None,
        attention_mask=None,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.transformer.attention_softmax(
        input_tensor, head_size=None, attention_mask=None, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
