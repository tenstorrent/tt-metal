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
        "shape": [[16, 3, 224, 224], [1, 1, 31, 31], [16, 3, 230, 224], [20, 3, 224, 256], [32, 64]],
        "padding": [
            ((0, 0), (0, 32), (0, 32)),  # +[0, 0, 32, 32]
            ((0, 0), (0, 15), (0, 31)),  # +[0, 0, 15, 31]
            ((0, 1), (3, 25), (32, 32)),
            ((0, 1), (3, 25), (4, 6)),
            ((0, 1), (3, 25), (4, 7)),
            ((0, 1), (0, 32), (0, 32)),
            ((1, 1), (2, 32), (0, 0)),
            [(32, 32)],
            ((0, 1), (0, 2)),
            ((1, 1), (4, 2)),
            ((0, 1), (0, 2)),
            ((1, 1), (4, 2)),
        ],
        "value": [0, 3],
        "dtype": [ttnn.int32, ttnn.uint32, ttnn.bfloat16],
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


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"
    if len(test_vector["shape"]) < len(test_vector["padding"]):
        return True, "Padding must be less than or equal to length of shape"
    front_padding = False
    for pad in test_vector["padding"]:
        if pad[0] > 0:
            front_padding = True
            break
    if front_padding and test_vector["layout"] == ttnn.TILE_LAYOUT:
        return True, "Front padding not supported with TILE_LAYOUT"

    return False, None


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

    torch_padding = []
    for i in range(len(padding) - 1, -1, -1):  # go through each dim of padding
        for p in padding[i]:
            torch_padding.append(p)  # each dim has 2 padding values
    padding = tuple(padding)

    # Measure performance of the embedding operation in ttnn
    start_time = start_measuring_time()

    ttnn_output_tensor = ttnn.pad(ttnn_input, padding=padding, value=value)

    e2e_perf = stop_measuring_time(start_time)

    torch_output_tensor = torch.nn.functional.pad(torch_input, torch_padding, mode="constant", value=value)

    # Convert the ttnn tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(torch_output_tensor, ttnn_output_tensor, 1.0)

    return [result, e2e_perf]
