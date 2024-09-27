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


# Create several shape types, such as [A, B, C, D] where any may or may not be included and any value ranges from 1-x
# for i -> rand(1,4):
#   shape should be [rand between 1,4096 for each N]
#   One index (0->n) should have length/value 1 to squeeze, index i #this should also be random
# dim = i or -(length-i) 50% chance here
# All can go in a for loop for how many samples you want
# This will do every combination of the following dtype and layout, so we will address RM and bfloat8 in invalidate vector
def generate_squeeze_config(num_samples=10):
    # Iterate to generate 'num_samples' configurations
    for _ in range(num_samples):
        # Randomly determine the number of dimensions (between 1 and 4)
        num_dims = random.randint(1, 4)

        # Generate random shape with dimensions between 1 and 4096
        shape = [random.randint(1, 256) for _ in range(num_dims)]

        # Select one dimension to be 1, so it can be squeezed
        squeeze_dim = random.randint(0, num_dims - 1)
        shape[squeeze_dim] = 1

        # Randomly determine whether the squeeze dimension index is positive or negative
        if random.random() < 0.5:
            dim = squeeze_dim  # positive dimension
        else:
            dim = squeeze_dim - num_dims  # negative dimension

        # Yield the configuration as a dictionary
        yield {
            "shape": shape,
            "dim": dim,  # This will either be positive or negative, randomly
        }


parameters = {
    "nightly": {
        # "squeeze_specs": list(generate_squeeze_config(num_samples=100)),
        "squeeze_specs": [
            {"shape": [1, 1, 25088], "dim": 0},
            {"shape": [1, 1, 480, 640], "dim": 1},
            {"shape": [1, 14, 1], "dim": -1},
            {"shape": [1, 19], "dim": 0},
            {"shape": [1, 25, 1], "dim": -1},
            {"shape": [1, 256, 1], "dim": -1},
            {"shape": [3, 1370, 1, 1, 1280], "dim": -2},
            {"shape": [3, 197, 1, 1, 1024], "dim": -2},
            {"shape": [3, 197, 1, 1, 768], "dim": -2},
            {"shape": [3, 50, 1, 1, 1024], "dim": -2},
            {"shape": [3, 50, 1, 1, 768], "dim": -2},
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
    squeeze_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the shape from squeeze_specs
    shape = squeeze_specs["shape"]
    dim = squeeze_specs.get("dim")  # Get the dimension to squeeze, if specified

    # Create random tensor of the specified shape
    tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)

    # Apply squeeze to the input tensor in PyTorch
    torch_output_tensor = tensor.squeeze(dim)

    # Convert the tensor to the ttnn tensor format
    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    # Measure performance of squeezing operation in ttnn
    start_time = start_measuring_time()

    # Apply squeeze in ttnn
    ttnn_output = ttnn.squeeze(ttnn_tensor, dim)

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensor back to a PyTorch tensor for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output)

    # Compare the results and return performance and accuracy check
    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999), e2e_perf]
