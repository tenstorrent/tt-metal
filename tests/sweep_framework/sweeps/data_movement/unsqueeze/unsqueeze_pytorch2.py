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
def generate_unsqueeze_config(num_samples=10):
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
        "unsqueeze_specs": [
            {"shape": [0], "dim": 1},
            {"shape": [1, 1, 1, 16], "dim": 4},
            {"shape": [1, 1, 10], "dim": 2},
            {"shape": [1, 1, 12], "dim": 2},
            {"shape": [1, 1, 14], "dim": 2},
            {"shape": [1, 1, 15], "dim": 2},
            {"shape": [1, 1, 16], "dim": 2},
            {"shape": [1, 1, 17], "dim": 1},
            {"shape": [1, 1, 17], "dim": 2},
            {"shape": [1, 1, 19], "dim": 2},
            {"shape": [1, 1, 1], "dim": 1},
            {"shape": [1, 1, 1], "dim": 2},
            {"shape": [1, 1, 201], "dim": 2},
            {"shape": [1, 1, 2048], "dim": 2},
            {"shape": [1, 1, 24], "dim": 2},
            {"shape": [1, 1, 256], "dim": 2},
            {"shape": [1, 1, 25], "dim": 2},
            {"shape": [1, 1, 2], "dim": 1},
            {"shape": [1, 1, 2], "dim": 2},
            {"shape": [1, 1, 32], "dim": 2},
            {"shape": [1, 1, 45], "dim": 2},
            {"shape": [1, 1, 46], "dim": 2},
            {"shape": [1, 1, 59], "dim": 2},
            {"shape": [1, 1, 5], "dim": 2},
            {"shape": [1, 1, 60], "dim": 2},
            {"shape": [1, 1, 6], "dim": 2},
            {"shape": [1, 1, 7], "dim": 2},
            {"shape": [1, 1, 8], "dim": 2},
            {"shape": [1, 1, 9], "dim": 2},
            {"shape": [1, 1, 1], "dim": 1},
            {"shape": [1, 1, 1], "dim": 2},
            {"shape": [1, 1, 0], "dim": 1},
            {"shape": [1, 1, 0], "dim": 2},
            {"shape": [1, 1, 1], "dim": 2},
            {"shape": [1, 10], "dim": 1},
            {"shape": [1, 12, 16, 2], "dim": 1},
            {"shape": [1, 120, 160], "dim": 1},
            {"shape": [1, 1280, 1], "dim": 3},
            {"shape": [1, 1280], "dim": 2},
            {"shape": [1, 12], "dim": 1},
            {"shape": [1, 14], "dim": 1},
            {"shape": [1, 15], "dim": 1},
            {"shape": [1, 17], "dim": 1},
            {"shape": [1, 19, 19], "dim": 1},
            {"shape": [1, 192], "dim": 1},
            {"shape": [1, 19], "dim": 1},
            {"shape": [1, 1], "dim": 1},
            {"shape": [1, 1], "dim": 2},
            {"shape": [1, 201], "dim": 1},
            {"shape": [1, 2048, 2048], "dim": 1},
            {"shape": [1, 2048], "dim": 1},
            {"shape": [1, 224, 224], "dim": 1},
            {"shape": [1, 23, 40], "dim": 3},
            {"shape": [1, 24], "dim": 1},
            {"shape": [1, 256], "dim": 0},
            {"shape": [1, 256], "dim": 1},
            {"shape": [1, 25], "dim": 1},
            {"shape": [1, 2], "dim": 1},
            {"shape": [1, 30, 40], "dim": 1},
            {"shape": [1, 32, 32], "dim": 1},
            {"shape": [1, 320, 1], "dim": 3},
            {"shape": [1, 320], "dim": 2},
            {"shape": [1, 32], "dim": 1},
            {"shape": [1, 384, 512], "dim": 1},
            {"shape": [1, 45, 45], "dim": 1},
            {"shape": [1, 45], "dim": 1},
            {"shape": [1, 46], "dim": 1},
            {"shape": [1, 5, 1, 16], "dim": 4},
            {"shape": [1, 5, 16], "dim": 2},
            {"shape": [1, 512], "dim": 1},
            {"shape": [1, 59, 59], "dim": 1},
            {"shape": [1, 59], "dim": 1},
            {"shape": [1, 5], "dim": 1},
            {"shape": [1, 60, 80], "dim": 1},
            {"shape": [1, 60], "dim": 1},
            {"shape": [1, 640, 1], "dim": 3},
            {"shape": [1, 640], "dim": 2},
            {"shape": [1, 64], "dim": 2},
            {"shape": [1, 6], "dim": 1},
            {"shape": [1, 7, 64], "dim": 1},
            {"shape": [1, 7, 7], "dim": 1},
            {"shape": [1, 720, 1280], "dim": 0},
            {"shape": [1, 7], "dim": 1},
            {"shape": [1, 8], "dim": 1},
            {"shape": [1, 9], "dim": 1},
            {"shape": [1, 1], "dim": 1},
            {"shape": [1, 0], "dim": 1},
            {"shape": [1, 1], "dim": 1},
            {"shape": [100, 256], "dim": 1},
            {"shape": [100], "dim": -1},
            {"shape": [10], "dim": 0},
            {"shape": [10], "dim": 1},
            {"shape": [12, 1, 1], "dim": 0},
            {"shape": [12, 10, 10], "dim": 0},
            {"shape": [12, 16, 2], "dim": 0},
            {"shape": [12, 197, 197], "dim": 0},
            {"shape": [12, 2, 2], "dim": 0},
            {"shape": [12, 49, 49], "dim": 0},
            {"shape": [12, 64, 64], "dim": 0},
            {"shape": [12, 1, 1], "dim": 0},
            {"shape": [120], "dim": 1},
            {"shape": [128], "dim": 1},
            {"shape": [12], "dim": -1},
            {"shape": [1370, 1, 3, 1280], "dim": 0},
            {"shape": [14], "dim": -1},
            {"shape": [15], "dim": 0},
            {"shape": [15], "dim": 1},
            {"shape": [16, 1, 1], "dim": 0},
            {"shape": [16, 1, 49, 49], "dim": 0},
            {"shape": [16, 1, 64, 64], "dim": 0},
            {"shape": [16, 10, 10], "dim": 0},
            {"shape": [16, 197, 197], "dim": 0},
            {"shape": [16, 2, 2], "dim": 0},
            {"shape": [16, 49, 49], "dim": 0},
            {"shape": [16, 49, 49], "dim": 1},
            {"shape": [16, 49], "dim": 1},
            {"shape": [16, 49], "dim": 2},
            {"shape": [16, 64, 64], "dim": 0},
            {"shape": [16, 64, 64], "dim": 1},
            {"shape": [16, 64], "dim": 1},
            {"shape": [16, 64], "dim": 2},
            {"shape": [16, 1, 1], "dim": 0},
            {"shape": [160], "dim": 0},
            {"shape": [16], "dim": -1},
            {"shape": [16], "dim": 1},
            {"shape": [17], "dim": 0},
            {"shape": [17], "dim": 1},
            {"shape": [19, 19], "dim": 0},
            {"shape": [197, 1, 3, 1024], "dim": 0},
            {"shape": [197, 1, 3, 768], "dim": 0},
            {"shape": [19], "dim": 0},
            {"shape": [1], "dim": 0},
            {"shape": [1], "dim": 1},
            {"shape": [2, 1, 7], "dim": 2},
            {"shape": [2, 7], "dim": 1},
            {"shape": [2048, 2048], "dim": 0},
            {"shape": [23], "dim": -1},
            {"shape": [24, 49, 49], "dim": 0},
            {"shape": [24, 64, 64], "dim": 0},
            {"shape": [240], "dim": 1},
            {"shape": [24], "dim": 0},
            {"shape": [24], "dim": 1},
            {"shape": [25], "dim": 1},
            {"shape": [28], "dim": -1},
            {"shape": [2], "dim": 0},
            {"shape": [2], "dim": 1},
            {"shape": [3, 1], "dim": 2},
            {"shape": [3, 320, 320], "dim": 0},
            {"shape": [3, 480, 640], "dim": 0},
            {"shape": [3, 49, 49], "dim": 0},
            {"shape": [3, 64, 64], "dim": 0},
            {"shape": [300], "dim": 1},
            {"shape": [30], "dim": 1},
            {"shape": [32, 32], "dim": 0},
            {"shape": [32, 49, 49], "dim": 0},
            {"shape": [32, 64, 64], "dim": 0},
            {"shape": [320], "dim": 1},
            {"shape": [3234], "dim": 1},
            {"shape": [32], "dim": -1},
            {"shape": [32], "dim": 0},
            {"shape": [3], "dim": 1},
            {"shape": [4, 1, 49, 49], "dim": 0},
            {"shape": [4, 1, 64, 64], "dim": 0},
            {"shape": [4, 49, 49], "dim": 0},
            {"shape": [4, 49, 49], "dim": 1},
            {"shape": [4, 49], "dim": 1},
            {"shape": [4, 49], "dim": 2},
            {"shape": [4, 64, 64], "dim": 0},
            {"shape": [4, 64, 64], "dim": 1},
            {"shape": [4, 64], "dim": 1},
            {"shape": [4, 64], "dim": 2},
            {"shape": [45, 45], "dim": 0},
            {"shape": [480], "dim": 1},
            {"shape": [50, 1, 3, 1024], "dim": 0},
            {"shape": [50, 1, 3, 768], "dim": 0},
            {"shape": [50], "dim": -1},
            {"shape": [56], "dim": -1},
            {"shape": [59, 59], "dim": 0},
            {"shape": [6, 1, 1], "dim": 0},
            {"shape": [6, 15, 15], "dim": 0},
            {"shape": [6, 17, 17], "dim": 0},
            {"shape": [6, 2, 2], "dim": 0},
            {"shape": [6, 49, 49], "dim": 0},
            {"shape": [6, 64, 64], "dim": 0},
            {"shape": [6, 1, 1], "dim": 0},
            {"shape": [60], "dim": 1},
            {"shape": [64, 1, 49, 49], "dim": 0},
            {"shape": [64, 1, 64, 64], "dim": 0},
            {"shape": [64, 49, 49], "dim": 1},
            {"shape": [64, 49], "dim": 1},
            {"shape": [64, 49], "dim": 2},
            {"shape": [64, 64, 64], "dim": 1},
            {"shape": [64, 64], "dim": 1},
            {"shape": [64, 64], "dim": 2},
            {"shape": [64], "dim": -1},
            {"shape": [64], "dim": 0},
            {"shape": [7, 7], "dim": 0},
            {"shape": [7], "dim": -1},
            {"shape": [7], "dim": 0},
            {"shape": [8, 1, 1], "dim": 0},
            {"shape": [8, 10, 10], "dim": 0},
            {"shape": [8, 2, 2], "dim": 0},
            {"shape": [8, 49, 49], "dim": 0},
            {"shape": [8, 64, 64], "dim": 0},
            {"shape": [8, 1, 1], "dim": 0},
            {"shape": [800], "dim": 1},
            {"shape": [8732], "dim": 1},
            {"shape": [96, 80], "dim": 0},
            {"shape": [1], "dim": 0},
            {"shape": [1], "dim": 1},
            {"shape": [0, 256], "dim": 0},
        ],
        "dtype": [ttnn.bfloat16],
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
    unsqueeze_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the shape from unsqueeze_specs
    shape = unsqueeze_specs["shape"]
    dim = unsqueeze_specs.get("dim")  # Get the dimension to unsqueeze, if specified

    # Create random tensor of the specified shape
    tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)

    # Apply unsqueeze to the input tensor in PyTorch
    torch_output_tensor = tensor.unsqueeze(dim)

    # Convert the tensor to the ttnn tensor format
    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    # Measure performance of unsqueezing operation in ttnn
    start_time = start_measuring_time()

    # Apply unsqueeze in ttnn
    ttnn_output = ttnn.unsqueeze(ttnn_tensor, dim)

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensor back to a PyTorch tensor for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output)

    # Compare the results and return performance and accuracy check
    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999), e2e_perf]
