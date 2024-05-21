# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn
import random
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

parameters = {
    "number_of_tensors": [1, 2, 3, 4, 5],
    "rank_of_tensors": [1, 2, 3, 4],
    "max_random_size_of_each_dim": [32],
    "dimension_to_concatenate_on": [0, 1, 2, 3, 4, 5],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "dtype": [ttnn.bfloat16],
    "memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
}


def skip(rank_of_tensors, layout, **_) -> Tuple[bool, Optional[str]]:
    if rank_of_tensors < 2 and layout == ttnn.TILE_LAYOUT:
        return True, "Tile layout is only supported for tensors with rank >= 2"
    return False, None


def xfail(number_of_tensors, rank_of_tensors, dimension_to_concatenate_on, **_) -> Tuple[bool, Optional[str]]:
    if number_of_tensors == 1:
        return True, "You must have at least two tensors to concat!"

    if dimension_to_concatenate_on >= rank_of_tensors:
        return (
            True,
            f"ttnn: Dimension out of range: dim {dimension_to_concatenate_on} cannot be used for tensors of rank {rank_of_tensors}",
        )

    return False, None


def run(
    number_of_tensors,
    rank_of_tensors,
    max_random_size_of_each_dim,
    dimension_to_concatenate_on,
    layout,
    dtype,
    memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    random.seed(0)

    def get_size_of_dim(index):
        size_of_dim = random.randint(1, max_random_size_of_each_dim)
        if layout == ttnn.ROW_MAJOR_LAYOUT and index == rank_of_tensors - 1 and size_of_dim % 2 == 1:
            size_of_dim = (size_of_dim + 1) % max_random_size_of_each_dim
            if size_of_dim == 0:
                size_of_dim = 2
        return size_of_dim

    def calculate_input_shape():
        return [get_size_of_dim(index) for index in range(rank_of_tensors)]

    input_shape = calculate_input_shape()
    torch_input_tensors = [torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)]

    if number_of_tensors > 1:
        first_tensor = torch_input_tensors[0]
        for _ in range(number_of_tensors - 1):
            shape = list(first_tensor.shape)
            if dimension_to_concatenate_on < rank_of_tensors:
                shape[dimension_to_concatenate_on] = get_size_of_dim(dimension_to_concatenate_on)
            new_tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)
            torch_input_tensors.append(new_tensor)

    input_tensors = [
        ttnn.from_torch(
            torch_input_tensor,
            device=device,
            layout=layout,
            dtype=dtype,
            memory_config=memory_config,
        )
        for torch_input_tensor in torch_input_tensors
    ]
    output_tensor = ttnn.concat(input_tensors, dim=dimension_to_concatenate_on)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_output_tensor = torch.concat(torch_input_tensors, dim=dimension_to_concatenate_on)
    return check_with_pcc(torch_output_tensor, output_tensor, 0.9999)
