# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from copy import deepcopy
import torch
import ttnn
import random
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


def dtype_to_rounding_mode(dtype):
    if dtype == ttnn.bfloat16:
        return 2
    elif dtype == ttnn.bfloat8_b:
        return 4
    return 1


def generate_configurations(
    number_of_tensors, rank_of_tensors, max_random_size, dimension_to_concatenate_on, layout, dtype
):
    base_shape = []

    base_shape = [
        random.randint(1, max_random_size) for _ in range(rank_of_tensors)
    ]  # all dims identical except for dim to concat on
    base_shape[dimension_to_concatenate_on] = -1

    variable_dim = [random.randint(1, max_random_size) for _ in range(number_of_tensors)]

    if layout == ttnn.ROW_MAJOR_LAYOUT:
        round_val = dtype_to_rounding_mode(dtype)
        if dimension_to_concatenate_on == rank_of_tensors - 1:
            for i in range(number_of_tensors):
                rem = variable_dim[i] % round_val
                if rem != 0:
                    variable_dim[i] = (variable_dim[i] + rem) % max_random_size
                    if variable_dim[i] == 0:
                        variable_dim[i] = round_val
        elif base_shape[-1] % round_val != 0:
            rem = base_shape[-1] % round_val
            base_shape[-1] = (base_shape[-1] + rem) % max_random_size
            if base_shape[-1] == 0:
                base_shape[-1] = round_val

    return base_shape, variable_dim


def generate_shapes(tensor_counts, ranks, layouts, dtypes, configs_per_variant=1):
    random.seed(0)

    shapes = []

    for _ in range(configs_per_variant):
        for rank in ranks:
            for layout in layouts:
                if rank < 2 and layout == ttnn.TILE_LAYOUT:
                    continue
                for dtype in dtypes:
                    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
                        continue
                    for concat_dim in range(rank):
                        for tensors in tensor_counts:
                            base_and_variable = generate_configurations(tensors, rank, 48, concat_dim, layout, dtype)
                            config = {
                                "tensors": tensors,
                                "rank": rank,
                                "concat_dim": concat_dim,
                                "base_shape": base_and_variable[0],
                                "variable_dim": base_and_variable[1],
                                "layout": layout,
                                "dtype": dtype,
                            }
                            shapes.append(config)

    return shapes


parameters = {
    "config": generate_shapes(
        [1, 2, 3, 4, 5], [1, 2, 3, 4], [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], [ttnn.bfloat16, ttnn.bfloat8_b], 3
    ),
    "memory_config": [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    ],
}


def run(config, memory_config, *, device) -> Tuple[bool, Optional[str]]:
    base_shape = config["base_shape"]
    variable_dim = config["variable_dim"]
    tensors = config["tensors"]
    rank = config["rank"]
    concat_dim = config["concat_dim"]
    layout = config["layout"]
    dtype = config["dtype"]

    torch_input_tensors = []

    for tensor in range(tensors):
        new_shape = deepcopy(base_shape)
        new_shape[concat_dim] = variable_dim[tensor]
        torch_input_tensors.append(torch_random(new_shape, -0.1, 0.1, dtype=torch.bfloat16))

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
    output_tensor = ttnn.concat(input_tensors, dim=concat_dim)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_output_tensor = torch.concat(torch_input_tensors, dim=concat_dim)
    if output_tensor.shape != torch_output_tensor.shape:
        return (
            False,
            f"Shapes do not match:  ttnn shape {output_tensor.shape} vs pytorch shape {torch_output_tensor.shape}",
        )
    return check_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def xfail(**_) -> Tuple[bool, Optional[str]]:
    return False, None
