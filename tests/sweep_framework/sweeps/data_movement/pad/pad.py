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
        "shape": [
            [16, 3, 224, 224],
            [1, 1, 31, 31],
        ],
        "padding": [((0, 0), (0, 32), (0, 32)), ((0, 0), (0, 15), (0, 31))],  # +[0, 0, 32, 32]  # +[0, 0, 15, 31]
        "value": [0],
        # "dtype": [ttnn.bfloat16],
        "dtype": [ttnn.uint32, ttnn.int32],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    }
}


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


def run(
    shape,
    padding,
    value,
    dtype,
    layout,
    *,
    device,
):
    torch_input = random_torch_tensor(dtype, shape)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=layout, dtype=dtype)

    # print(padding)
    torch_padding = []
    for i in range(len(padding) - 1, -1, -1):  # go through each dim of padding
        for p in padding[i]:
            torch_padding.append(p)  # each dim has 2 padding values
    # print(torch_padding)

    # Measure performance of the embedding operation in ttnn
    start_time = start_measuring_time()

    ttnn_output_tensor = ttnn.pad(ttnn_input, padding=padding, value=value)

    e2e_perf = stop_measuring_time(start_time)

    torch_output_tensor = torch.nn.functional.pad(torch_input, torch_padding, mode="constant", value=value)

    # Convert the ttnn tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
