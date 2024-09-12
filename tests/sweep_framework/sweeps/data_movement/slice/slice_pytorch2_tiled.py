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

parameters = {
    "nightly": {
        "slice_specs": [
            {"dims": [1, 4], "dim": 1, "start": 0, "end": -1, "step": 4},
            {"dims": [1, 1, 1, 10], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 15], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 19], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 2048], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 2048], "dim": 3, "start": 0, "end": 46},
            {"dims": [1, 1, 1, 2048], "dim": 3, "start": 0, "end": 6},
            {
                "dims": [1, 1, 1, 2048],
                "dim": 3,
                "start": 0,
                "end": None,  # Represented as None since it's a variable (s10 + 1)
            },
            {"dims": [1, 1, 1, 24], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 256], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 25], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 45], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 5], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 7], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1, 8], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 1024, 1024], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 1, 12, 16, 2], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 16, 32], "dim": 3, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 1, 16, 64], "dim": 3, "start": 0, "end": 32},
            {"dims": [1, 1, 19, 19], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 1, 1], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 1, 2048, 2048], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 1, 2048, 2048], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 1, 2048, 2048], "dim": 2, "start": 0, "end": 5},
            {
                "dims": [1, 1, 2048, 2048],
                "dim": 2,
                "start": None,  # Represented as None since it's a variable
                "end": None,  # Represented as None since it's a variable (s10 + 1)
            },
            {"dims": [1, 1, 32, 32], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 1, 32], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 1, 384, 512], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 1, 40], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 1, 45, 2048], "dim": 3, "start": 0, "end": 45},
            {"dims": [1, 1, 45, 45], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 1, 5, 2048], "dim": 3, "start": 0, "end": 5},
            {"dims": [1, 1, 7, 1024], "dim": 3, "start": 0, "end": 7},
            {"dims": [1, 1, 7, 64], "dim": 3, "start": 32, "end": -1},
            {"dims": [1, 1, 7, 7], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 100, 192], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 10], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 12, 2, 2], "dim": 2, "start": -1, "end": -1},
            {"dims": [1, 1370, 1280], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 14, 28, 192], "dim": 2, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 14, 28, 256], "dim": 2, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 1445, 192], "dim": 1, "start": -100, "end": -1},
            {"dims": [1, 145, 768], "dim": 1, "start": 1, "end": -1},
            {"dims": [1, 15], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 16, 112, 112], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 16, 2, 2], "dim": 2, "start": -1, "end": -1},
            {"dims": [1, 16, 32, 192], "dim": 2, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 16, 32, 256], "dim": 2, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 1876, 768], "dim": 1, "start": 0, "end": 1},
            {"dims": [1, 1876, 768], "dim": 1, "start": 0, "end": None},  # Represented as None since it's a variable
            {"dims": [1, 192], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 196, 1024], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 196, 768], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 197, 1024], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 197, 768], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 19], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 1], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 2, 30, 40], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 2048], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 23, 40, 128], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 23, 40, 128], "dim": 3, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 23, 40], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 24, 56, 56], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 24], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 256], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 28, 28, 128], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 28, 28, 192], "dim": 1, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 28, 28, 256], "dim": 1, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 28, 28, 96], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 28, 56, 128], "dim": 2, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 28, 56, 96], "dim": 2, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 3, 224, 224], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 30, 40], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 32, 32, 128], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 32, 32, 128], "dim": 3, "start": 0, "end": 64},
            {"dims": [1, 32, 32, 192], "dim": 1, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 32, 32, 256], "dim": 1, "start": 1, "end": -1, "step": 2},
            {"dims": [1, 32, 32, 96], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 32, 64, 128], "dim": 2, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 32, 64, 96], "dim": 2, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 32], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 40], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 4251, 192], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 45], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 46], "dim": 1, "start": 45, "end": -1},
            {"dims": [1, 48, 112, 112], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 5, 16, 32], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 5, 16, 32], "dim": 3, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 5, 16, 64], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 50, 1024], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 50, 768], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 512], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 512], "dim": 1, "start": 0, "end": 256},
            {"dims": [1, 512], "dim": 1, "start": 0, "end": 8},
            {"dims": [1, 514], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 56, 56, 128], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 56, 56, 128], "dim": 1, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 56, 56, 96], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 56, 56, 96], "dim": 1, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 5], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 6, 2, 2], "dim": 2, "start": -1, "end": -1},
            {"dims": [1, 60, 80], "dim": 2, "start": 0, "end": -1},
            {"dims": [1, 600, 768], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 64, 64, 128], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 64, 64, 128], "dim": 1, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 64, 64, 96], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 64, 64, 96], "dim": 1, "start": 0, "end": -1, "step": 2},
            {"dims": [1, 64], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 6], "dim": 1, "start": 5, "end": -1},
            {"dims": [1, 7, 71, 64], "dim": 3, "start": 0, "end": -1},
            {"dims": [1, 7, 73, 64], "dim": 2, "start": 0, "end": -2},
            {"dims": [1, 71, 7, 64], "dim": 3, "start": 0, "end": 32},
            {"dims": [1, 768], "dim": 1, "start": 0, "end": -1},
            {"dims": [1, 77], "dim": 1, "start": 0, "end": 7},
            {"dims": [1, 7], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 8, 2, 2], "dim": 2, "start": -1, "end": -1},
            {"dims": [1, 80, 3000], "dim": 0, "start": 0, "end": -1},
            {"dims": [1, 8], "dim": 0, "start": 0, "end": -1},
            # {
            #     "dims": [1, "s0"],
            #     "dim": 1,
            #     "start": -1,
            #     "end": -1
            # },
            # {
            #     "dims": [1, "s1 + 1"],
            #     "dim": 1,
            #     "start": None,  # Represented as None since it's a variable
            #     "end": -1
            # },
            {"dims": [2, 1, 1, 7], "dim": 3, "start": 0, "end": -1},
            {"dims": [2048, 64], "dim": 0, "start": 0, "end": 7},
            {"dims": [3234, 4], "dim": 1, "start": 0, "end": -1, "step": 4},
            {"dims": [3234, 4], "dim": 1, "start": 0, "end": 2},
            {"dims": [3], "dim": 0, "start": 0, "end": -1},
            {"dims": [448, 768], "dim": 0, "start": 0, "end": 1},
            {
                "dims": [448, 768],
                "dim": 0,
                "start": None,  # Represented as None since it's a variable
                "end": None,  # Represented as None since it's a variable (s2 + 1)
            },
            {"dims": [8732, 4], "dim": 1, "start": 0, "end": -1, "step": 4},
            {"dims": [8732, 4], "dim": 1, "start": 0, "end": 2},
        ],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.TILE_LAYOUT],
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
    device.enable_async(False)

    dims = slice_specs["dims"]
    dim = slice_specs["dim"]
    start = slice_specs["start"]
    end = slice_specs["end"]
    step = slice_specs.get("step", 1)

    tensor = torch_random(dims, -0.1, 0.1, dtype=torch.bfloat16)
    # Create a slice object
    slice_obj = slice(start, end, step)

    # Prepare indices for slicing in the specified dimension
    indices = [slice(None)] * len(dims)  # By default, select all elements along every dimension
    indices[dim] = slice_obj  # Apply slicing to the target dimension
    indices = tuple(indices)

    # Apply slicing to the input_tensor
    torch_output_tensor = tensor[indices]

    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    start_time = start_measuring_time()
    ttnn_output = ttnn_tensor[indices]
    e2e_perf = stop_measuring_time(start_time)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output)
    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999), e2e_perf]
