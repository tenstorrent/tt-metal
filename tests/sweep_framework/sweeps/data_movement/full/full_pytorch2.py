# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
random.seed(0)


def generate_fill_specs(n, x, y):
    fill_specs = []

    for _ in range(n):
        # Randomly sample values for the first two dimensions (batch and channel size)
        batch_size = random.randint(1, 128)
        channel_size = random.randint(1, 128)

        # Randomly sample values for height and width, with a max bound of x
        height = random.randint(1, x)
        width = random.randint(1, x)

        # Randomly sample a fill value between -y and y
        fill_value = random.uniform(-y, y)

        # Append a new dictionary with the sampled values
        fill_specs.append({"shape": [batch_size, channel_size, height, width], "fill_value": fill_value})

    return fill_specs


parameters = {
    "nightly": {
        "fill_specs": generate_fill_specs(100, 128, 500),
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
    if test_vector["dtype"] == ttnn.bfloat8_b:
        if len(test_vector["slice_specs"]["dims"]) < 2:
            return True, "bfloat8_b not supported with dims  < 2"

    return False, None


def run(
    fill_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the shape and fill_value from the test case
    shape = fill_specs["shape"]
    fill_value = fill_specs["fill_value"]

    # Create a tensor filled with `fill_value` using torch.full
    torch_tensor = torch.full(shape, fill_value, dtype=torch.float32)

    # Measure performance of the full operation in `ttnn`
    start_time = start_measuring_time()

    # Apply the `ttnn.full` operation
    ttnn_filled_tensor = ttnn.full(shape, fill_value, device, memory_config=None)

    e2e_perf = stop_measuring_time(start_time)

    # Convert the `ttnn` tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_filled_tensor)

    # Compare the PyTorch and `ttnn` tensors
    result = check_with_pcc(torch_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
