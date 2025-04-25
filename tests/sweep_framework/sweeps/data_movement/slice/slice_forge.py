# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import json
import os
import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
random.seed(0)

# Load the processed slice specs from slice_forge_processed.json
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "slice_forge_processed.json")
with open(json_path, "r") as f:
    processed_slice_specs = json.load(f)

parameters = {
    "nightly": {
        "slice_specs": processed_slice_specs,
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
    slice_specs,
    dtype,
    layout,
    *,
    device,
):
    dims = slice_specs["dims"]
    begins = slice_specs["begins"]
    ends = slice_specs["ends"]
    steps = slice_specs["step"]

    # Create the torch input tensor
    tensor = torch_random(dims, -0.1, 0.1, dtype=torch.bfloat16)

    # Construct Python slice objects from begins, ends, steps
    indices = [slice(begins[i], ends[i], steps[i]) for i in range(len(begins))]

    # Apply slicing to the torch tensor
    torch_output_tensor = tensor[tuple(indices)]

    # Convert the input tensor to TTNN
    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    # Run the slicing on TTNN
    start_time = start_measuring_time()
    ttnn_output = ttnn.slice(ttnn_tensor, begins, ends, steps)
    e2e_perf = stop_measuring_time(start_time)

    # Convert TTNN output back to torch
    ttnn_output_tensor = ttnn.to_torch(ttnn_output)

    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999), e2e_perf]
