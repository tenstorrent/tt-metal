# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
# seed for random
random.seed(0)

parameters = {
    "nightly": {
        "shape": [
            (1, 16),
            (16, 1),
            (1, 17),
            (17, 1),
            (16, 16),
            (17, 17),
            (31, 31),
            (33, 33),
            (65, 65),
            (97, 97),
            (1, 2, 3, 2, 1, 2, 97, 97),
            (1, 32),
            (16, 32),
            (1, 32),
            (17, 32),
            (16, 32),
            (17, 32),
            (31, 32),
            (33, 64),
            (65, 64),
            (97, 96),
            (1, 2, 3, 2, 1, 2, 97, 96),
        ],
        "fill_value": [1, float("-inf"), float("inf"), 2.5],
        "dtype": [ttnn.uint32, ttnn.bfloat16, ttnn.bfloat8_b],
    }
}


def create_nd_padded_tiled_tensor(shape, tile_size, fill_value, dtype):
    """
    Creates a tensor with shape `shape` with random values, and another tensor with the same values,
    but with the last 2 dimensions padded to the nearest multiple of tile_size using fill_value.

    Args:
        shape (tuple): Shape of the original tensor.
        tile_size (int): Size to which the last two dimensions will be padded.
        fill_value (float or int): Value used for padding.
        dtype (torch.dtype): Data type of the tensors.

    Returns:
        tuple: A tuple containing the original tensor and the padded tensor.
    """
    # Create a tensor with random values
    if dtype == torch.float32:
        tensor = torch_random(shape, -15.0, 15.0, dtype=dtype)
    else:
        tensor = torch.randint(0, 10, shape, dtype=dtype)

    # Calculate the padded sizes for the last two dimensions
    padded_shape = list(shape)
    padded_shape[-2] = (padded_shape[-2] + tile_size - 1) // tile_size * tile_size
    padded_shape[-1] = (padded_shape[-1] + tile_size - 1) // tile_size * tile_size

    # Create a padded tensor filled with fill_value
    padded_tensor = torch.full(padded_shape, fill_value, dtype=dtype)

    # Copy the original tensor into the padded tensor
    padded_tensor[..., : shape[-2], : shape[-1]] = tensor

    return tensor, padded_tensor


ttnn_dtype_to_torch_dtype = {
    ttnn.uint32: torch.int32,
    ttnn.bfloat16: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["dtype"] == ttnn.uint32:
        if isinstance(test_vector["fill_value"], float):
            return True, "int dtype should have int fill_value"
    if test_vector["dtype"] == ttnn.bfloat8_b:
        if test_vector["shape"][-1] % 32 != 0:
            return True, "bfloat8_b not supported with last dim not width aligned to tile width"

    return False, None


def run(
    shape,
    fill_value,
    dtype,
    *,
    device,
):
    device.enable_async(False)

    torch.manual_seed(1234)
    if isinstance(fill_value, str):
        if fill_value.lower() == "inf":
            fill_value = float("inf")
        elif fill_value.lower() == "-inf":
            fill_value = float("-inf")
        else:
            raise ValueError(f"Unexpected string fill_value: {fill_value}")

    torch_input_tensor, padded_torch_tensor = create_nd_padded_tiled_tensor(
        shape, 32, fill_value, ttnn_dtype_to_torch_dtype[dtype]
    )
    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    start_time = start_measuring_time()

    output_tensor = ttnn.fill_implicit_tile_padding(input_tensor, fill_value, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    e2e_perf = stop_measuring_time(start_time)

    padded_torch_output_tensor = ttnn.from_device(output_tensor).to_torch_with_padded_shape()

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(padded_torch_tensor, padded_torch_output_tensor)

    return [result, e2e_perf]
