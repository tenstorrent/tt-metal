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
        "repeat_specs": [
            {"shape": [1, 1, 1], "repeats": [1, 1, 1]},
            {"shape": [1, 1, 2048, 2048], "repeats": [1, 1, 1, 1]},
            {"shape": [1, 1, 256], "repeats": [1, 1, 1]},
            {"shape": [1, 128, 256], "repeats": [1, 1, 1]},
            {"shape": [100, 1, 256], "repeats": [1, 1, 1]},
            {"shape": [4, 2], "repeats": [1, 1]},
            {"shape": [4, 2], "repeats": [1444, 1]},
            {"shape": [4, 2], "repeats": [9, 1]},
            {"shape": [6, 2], "repeats": [1, 1]},
            {"shape": [6, 2], "repeats": [100, 1]},
            {"shape": [6, 2], "repeats": [25, 1]},
            {"shape": [6, 2], "repeats": [361, 1]},
            {"shape": [6, 2], "repeats": [4, 1]},
            {"shape": [6, 2], "repeats": [400, 1]},
            {"shape": [6, 2], "repeats": [9, 1]},
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
    repeat_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the shape and repeat dimensions from repeat_specs
    shape = repeat_specs["shape"]
    repeat_dims = repeat_specs["repeats"]  # Number of repetitions for each dimension

    # Create a random tensor of the specified shape
    tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)

    # Apply repeat using PyTorch's repeat function
    torch_output_tensor = tensor.repeat(*repeat_dims)

    # Convert the tensor to the ttnn tensor format
    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    # Measure performance of the repeat operation in ttnn
    start_time = start_measuring_time()

    # Apply repeat in ttnn
    ttnn_output_tensor = ttnn.repeat(ttnn_tensor, repeat_dims)

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
