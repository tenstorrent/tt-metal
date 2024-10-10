# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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
        "split_specs": [
            {"shape": [1, 1, 32], "split_size": 16, "dim": -1},
            {"shape": [1, 1, 4, 768], "split_size": 256, "dim": -1},
            {"shape": [1, 1024, 5120], "split_size": 2560, "dim": -1},
            {"shape": [1, 14, 2], "split_size": 1, "dim": -1},
            {"shape": [1, 25, 2], "split_size": 1, "dim": -1},
            {"shape": [1, 256, 10240], "split_size": 5120, "dim": -1},
            {"shape": [1, 256, 2], "split_size": 1, "dim": -1},
            {"shape": [1, 4096, 2560], "split_size": 1280, "dim": -1},
            {"shape": [1, 5, 32], "split_size": 16, "dim": -1},
            {"shape": [1, 5, 4, 768], "split_size": 256, "dim": -1},
            {"shape": [1, 64, 10240], "split_size": 5120, "dim": -1},
            {"shape": [1, 7, 2304], "split_size": 768, "dim": 2},
            {"shape": [768, 256], "split_size": 256, "dim": -1},
            {"shape": [768], "split_size": 256, "dim": -1},
        ],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"

    return False, None


def run(
    split_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the shape and split_size from split_specs
    shape = split_specs["shape"]
    dim = split_specs.get("dim", 0)  # Get the dimension to split, default to 0 if unspecified
    split_size = split_specs["split_size"]  # Number of splits

    # Create a random tensor of the specified shape
    tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)

    # Apply split using torch's chunk function
    torch_output_tensors = torch.chunk(tensor, split_size, dim)

    # Convert the tensor to the ttnn tensor format
    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    # Measure performance of the split operation in ttnn
    start_time = start_measuring_time()

    # Apply split in ttnn
    ttnn_output_tensors = ttnn.split(ttnn_tensor, split_size, dim)

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensors back to PyTorch tensors for comparison
    # Convert the ttnn output tensors back to PyTorch tensors
    ttnn_output_tensors = [ttnn.to_torch(tt) for tt in ttnn_output_tensors]

    # Concatenate the PyTorch tensors from both torch.chunk and ttnn.split along the split dimension
    torch_concat_output = torch.cat(torch_output_tensors, dim)
    ttnn_concat_output = torch.cat(ttnn_output_tensors, dim)

    # Compare the concatenated tensors and return performance and accuracy check
    result = check_with_pcc(torch_concat_output, ttnn_concat_output, 0.999)

    return [result, e2e_perf]
