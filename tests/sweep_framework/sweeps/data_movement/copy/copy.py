# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn
import sys

from typing import Optional, Tuple

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 15

parameters = {
    "nightly": {
        "shape": [
            [1, 1, 1, 10],
            [1, 1, 1, 12],
            [1, 1, 1, 14],
            [1, 1, 1, 15],
            [1, 1, 1, 201],
            [1, 1, 1, 2048],
            [1, 1, 1, 24],
            [1, 1, 1, 25],
            [1, 1, 1, 42],
            [1, 1, 1, 46],
            [1, 1, 1, 5],
            [1, 1, 1, 60],
            [1, 1, 1, 6],
            [1, 1, 1, 7],
            [1, 1, 1, 9],
            [1, 1, 12, 16],
            [1, 1, 19, 19],
            [1, 1, 19, 19],
            [1, 1, 1],
            [1, 1, 23, 40],
            [1, 1, 24, 24],
            [1, 1, 24, 24],
            [1, 1, 32, 1],
            [1, 1, 32, 32],
            [1, 1, 384, 512],
            [1, 1, 45, 45],
            [1, 1, 59, 59],
            [1, 1, 720, 1280],
            [1, 10],
            [1, 16, 32, 32],
            [1, 1],
            [1, 3, 224, 224],
            [1, 7],
            [10, 10],
            [100],
            [1066],
            [120],
            [128],
            [12],
            [136],
            [14],
            [15, 15],
            [160],
            [16],
            [17, 17],
            [2, 1, 7, 7],
            [2, 2],
            [2, 7],
            [23],
            [240],
            [25],
            [28],
            [300],
            [30],
            [320],
            [32],
            [40],
            [480],
            [50],
            [56],
            [60],
            [640],
            [64],
            [68],
            [7, 7],
            [7],
            [800],
            [80],
            [],
        ],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
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
    input_tensor = torch_random(shape, 0, 256, torch_dtype)
    input_tensor_tt = ttnn.from_torch(input_tensor, dtype=dtype, layout=layout, device=device)

    output_tensor = torch.zeros(shape, dtype=torch_dtype)
    output_tensor_tt = ttnn.from_torch(output_tensor, dtype=dtype, layout=layout, device=device)

    start_time = start_measuring_time()
    ttnn.copy(input_tensor_tt, output_tensor_tt)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = output_tensor_tt.cpu().to_torch()

    return [check_with_pcc(input_tensor, output_tensor, 0.999), e2e_perf]
