# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn

from typing import Optional, Tuple

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 15  # longer timeout since permute calls transpose recursively
random.seed(0)

parameters = {
    "nightly": {
        "transpose_specs": [
            {"shape": [1, 16, 256, 64], "dim0": -1, "dim1": -2},
            {"shape": [1, 16, 256, 64], "dim0": 2, "dim1": 3},
            {"shape": [1024, 1024], "dim0": -1, "dim1": -2},
            {"shape": [1024, 4096], "dim0": -1, "dim1": -2},
            {"shape": [2, 1024], "dim0": -1, "dim1": -2},
            {"shape": [4096, 1024], "dim0": -1, "dim1": -2},
            {"shape": [1024, 1024], "dim0": 0, "dim1": 1},
            {"shape": [1024, 4096], "dim0": 0, "dim1": 1},
            {"shape": [2, 1024], "dim0": 0, "dim1": 1},
            {"shape": [4096, 1024], "dim0": 0, "dim1": 1},
            {"shape": [1, 32, 12, 100], "dim0": -2, "dim1": -3},
        ],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    }
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"

    return False, None


def run(
    transpose_specs,
    dtype,
    layout,
    *,
    device,
):
    torch_input_tensor = torch_random(
        transpose_specs["shape"], -0.1, 0.1, dtype=torch.bfloat16
    )  # returns to torch tensor
    torch_output_tensor = torch.transpose(torch_input_tensor, transpose_specs["dim0"], transpose_specs["dim1"])

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=dtype, layout=layout)

    start_time = start_measuring_time()
    ttnn_output = ttnn.transpose(ttnn_input_tensor, transpose_specs["dim0"], transpose_specs["dim1"])
    e2e_perf = stop_measuring_time(start_time)

    ttnn_output_tensor = ttnn.to_torch(ttnn_output)
    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.9999), e2e_perf]
