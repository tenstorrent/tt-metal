# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn
import random
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

parameters = {
    "make_repeat_a_tensor": [False, True],
    "rank_of_tensor": [1, 2, 3, 4],
    "max_random_size_of_each_dim": [32],
    "dimension_to_repeat_on": [0, 1, 2, 3, 4, 5],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "dtype": [ttnn.bfloat16],
    "memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
}


def skip(make_repeat_a_tensor, rank_of_tensor, max_random_size_of_each_dim, layout, **_) -> Tuple[bool, Optional[str]]:
    if rank_of_tensor < 2 and layout == ttnn.TILE_LAYOUT:
        return True, "Tile layout is only supported for tensors with rank >= 2"
    return False, None


def is_expected_to_fail(
    make_repeat_a_tensor, rank_of_tensor, max_random_size_of_each_dim, dimension_to_repeat_on, **_
) -> Tuple[bool, Optional[str]]:
    # The following conditions represent what the error message should start with
    # in those cases where there was a failure.
    dimension_range = f"[{-rank_of_tensor}, {rank_of_tensor - 1}]"
    if not make_repeat_a_tensor and dimension_to_repeat_on >= rank_of_tensor:
        return (
            True,
            f"TTNN: Dimension out of range (expected to be in range of {dimension_range}, but got {dimension_to_repeat_on})",
        )
    if make_repeat_a_tensor and dimension_to_repeat_on < rank_of_tensor:
        return True, "TTNN: repeats must have the same size as input along dim"
    if make_repeat_a_tensor and dimension_to_repeat_on >= rank_of_tensor:
        return (
            True,
            f"TTNN: Dimension out of range (expected to be in range of {dimension_range}, but got {dimension_to_repeat_on})",
        )
    return False, None


def run(
    make_repeat_a_tensor,
    rank_of_tensor,
    max_random_size_of_each_dim,
    dimension_to_repeat_on,
    layout,
    dtype,
    memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    random.seed(0)

    def get_size_of_dim(index):
        size_of_dim = random.randint(1, max_random_size_of_each_dim)
        if layout == ttnn.ROW_MAJOR_LAYOUT and index == rank_of_tensor - 1 and size_of_dim % 2 == 1:
            size_of_dim = (size_of_dim + 1) % max_random_size_of_each_dim
            if size_of_dim == 0:
                size_of_dim = 2
        return size_of_dim

    def calculate_input_shape():
        return [get_size_of_dim(index) for index in range(rank_of_tensor)]

    input_shape = calculate_input_shape()
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)

    if make_repeat_a_tensor:
        torch_repeat = torch.tensor([get_size_of_dim(index) for index in range(rank_of_tensor)])
        # for now, don't require the repeat tensor to be on device
        repeat = ttnn.from_torch(torch_repeat, layout=layout, dtype=dtype)
    else:
        torch_repeat = random.randint(1, max_random_size_of_each_dim)
        repeat = torch_repeat

    input_tensors = ttnn.from_torch(
        torch_input_tensor, device=device, layout=layout, dtype=dtype, memory_config=memory_config
    )
    output_tensor = ttnn.repeat_interleave(input_tensors, repeat, dim=dimension_to_repeat_on)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_output_tensor = torch.repeat_interleave(torch_input_tensor, torch_repeat, dim=dimension_to_repeat_on)
    return check_with_pcc(torch_output_tensor, output_tensor, 0.9999)
