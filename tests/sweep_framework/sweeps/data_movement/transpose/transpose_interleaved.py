# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn

from typing import Optional, Tuple

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 20  # longer timeout since permute calls transpose recursively
random.seed(0)


def generate_transpose_shape(num_samples):
    for _ in range(num_samples):
        shape = [random.randint(1, 96) for _ in range(4)]
        yield shape


parameters = {
    "interleaved_4d": {
        "shape": list(generate_transpose_shape(8)),
        "dim0": [-4, -3, -2, -1, 0, 1, 2, 3],
        "dim1": [-4, -3, -2, -1, 0, 1, 2, 3],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    }
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"
    return False, None


def run(
    shape,
    dim0,
    dim1,
    layout,
    dtype,
    *,
    device,
):
    torch_input_tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)  # returns to torch tensor
    torch_output_tensor = torch.transpose(torch_input_tensor, dim0, dim1)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=dtype, layout=layout)

    start_time = start_measuring_time()
    ttnn_output = ttnn.transpose(ttnn_input_tensor, dim0, dim1)
    e2e_perf = stop_measuring_time(start_time)

    ttnn_output_tensor = ttnn.to_torch(ttnn_output)
    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.9999), e2e_perf]
