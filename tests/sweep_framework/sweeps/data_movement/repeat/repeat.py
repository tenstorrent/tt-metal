# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn
import sys

from typing import Optional, Tuple

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from functools import reduce

TIMEOUT = 15

dtypes = [ttnn.bfloat16]
shapes = [(1, 2, 4, 4), (1, 1, 1, 1)] + [(1, 2, 2 * j, 4 * j) for j in range(2, 12)]
repeat_specs = [(2, 2, 2, 2), (1, 2, 1, 1)] + [(1, 3, 2 * j, 4 * j) for j in range(2, 12)]

parameters = {
    "nightly": {
        "shape": shapes,
        "repeat_shape": repeat_specs,
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "dtype": dtypes,
    }
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"
        if test_vector["dtype"] == ttnn.bfloat16 and len(test_vector["shape"]) > 1 and test_vector["shape"][-1] < 2:
            return True, "bfloat16 in ROW_MAJOR_LAYOUT not supported with inner dim < 2"
    return False, None


def run(
    shape,
    repeat_shape,
    layout,
    dtype,
    *,
    device,
):
    torch_dtype = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
    }[dtype]

    mul = lambda x, y: x * y
    torch_input_tensor = torch.arange(0, reduce(mul, shape, 1), dtype=torch_dtype).reshape(shape)

    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)

    start_time = start_measuring_time()
    tt_result = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    e2e_perf = stop_measuring_time(start_time)
    tt_result = tt_result.cpu().to_torch()

    assert (
        tt_result.shape == torch_result.shape
    ), f"TT result shape {tt_result.shape} does not match torch shape {torch_result.shape}"

    return [check_with_pcc(torch_result, tt_result, 0.999), e2e_perf]
