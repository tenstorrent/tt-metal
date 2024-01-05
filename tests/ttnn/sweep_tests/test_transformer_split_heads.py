# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from tests.ttnn.sweep_tests.sweep import sweep_or_reproduce
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import skip_for_wormhole_b0, torch_random


def test_sweep(device, sweep_index):
    parameters = {
        "batch_size": [1],
        "sequence_size": [384, 1024],
        "num_heads": [4, 16],
        "head_size": [64, 128],
        "order": [(0, 2, 1, 3), (0, 2, 3, 1)],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    }

    def skip(**_):
        return False

    def run(
        batch_size,
        num_heads,
        sequence_size,
        head_size,
        order,
        input_dtype,
        input_memory_config,
    ):
        input_shape = (batch_size, sequence_size, num_heads * head_size)
        torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
        torch_output_tensor = ttnn.transformer._torch_split_heads(torch_input_tensor, num_heads=num_heads, order=order)

        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            device=device,
            dtype=input_dtype,
            memory_config=input_memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        output_tensor = ttnn.transformer.split_heads(input_tensor, num_heads=num_heads, order=order)
        output_tensor = ttnn.to_torch(output_tensor)

        return check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    sweep_or_reproduce(__file__, run, skip, parameters, sweep_index)
