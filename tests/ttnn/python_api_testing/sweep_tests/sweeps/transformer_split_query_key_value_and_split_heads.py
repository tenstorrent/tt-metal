# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_size": [1],
    "sequence_size": [384, 1024],
    "num_heads": [4, 16],
    "head_size": [64, 128],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}


def skip(**_):
    return False


def run(batch_size, num_heads, sequence_size, head_size, input_dtype, input_memory_config, *, device):
    input_shape = (batch_size, sequence_size, num_heads * head_size * 3)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = ttnn.transformer._torch_split_query_key_value_and_split_heads(torch_input_tensor, num_heads=num_heads)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor, num_heads=num_heads
    )
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.to_torch(value_tensor)

    query_matches, query_message = check_with_pcc(torch_query_tensor, query_tensor, 0.999)
    key_matches, key_message = check_with_pcc(torch_key_tensor, key_tensor, 0.999)
    value_matches, value_message = check_with_pcc(torch_value_tensor, value_tensor, 0.999)

    passed = query_matches and key_matches and value_matches
    message = ""
    if not query_matches:
        message += f"query: {query_message}; "
    if not key_matches:
        message += f"key: {key_message}; "
    if not value_matches:
        message += f"value: {value_message}; "

    return passed, message
