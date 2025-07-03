# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

TIMEOUT = 15

parameters = {
    "default": {
        "params": [
            ((1, 1, 16384, 256), -1, False),
            ((1, 1, 19200, 300), -1, False),
            ((1, 12, 1, 10), -1, False),
            ((1, 12, 1, 1), -1, False),
            ((1, 12, 1, 2), -1, False),
            ((1, 12, 1, 46), -1, False),
            ((1, 12, 1, 100 + 1), -1, False),
            ((1, 12, 1, 1000 + 1), -1, False),
            ((1, 12, 10, 10), -1, False),
            ((1, 12, 12, 12), -1, False),
            ((1, 12, 14, 14), -1, False),
            ((1, 12, 16, 16), -1, False),
            ((1, 12, 197, 197), -1, False),
            ((1, 12, 25, 25), -1, False),
            ((1, 12, 45, 45), -1, False),
            ((1, 12, 7, 7), -1, False),
            ((1, 12, 9, 9), -1, False),
            ((1, 16, 1, 10), -1, False),
            ((1, 16, 1, 1), -1, False),
            ((1, 16, 1, 2), -1, False),
            ((1, 16, 1, 6), -1, False),
            ((1, 16, 1, 100 + 1), -1, False),
            ((1, 16, 1, 1000 + 1), -1, False),
            ((1, 16, 10, 10), -1, False),
            ((1, 16, 197, 197), -1, False),
            ((1, 16, 256, 256), -1, False),
            ((1, 16, 32, 32), -1, False),
            ((1, 16, 5, 5), -1, False),
            ((1, 16, 9, 9), -1, False),
            ((1, 2, 4096, 256), -1, False),
            ((1, 2, 4800, 300), -1, False),
            ((1, 24, 49, 49), -1, False),
            ((1, 24, 64, 64), -1, False),
            ((1, 3, 1445, 1445), -1, False),
            ((1, 32, 49, 49), -1, False),
            ((1, 32, 64, 64), -1, False),
            ((1, 5, 1024, 256), -1, False),
            ((1, 5, 1200, 300), -1, False),
            ((1, 6, 1, 15), -1, False),
            ((1, 6, 1, 17), -1, False),
            ((1, 6, 1, 1), -1, False),
            ((1, 6, 1, 2), -1, False),
            ((1, 6, 1, 100 + 1), -1, False),
            ((1, 6, 15, 15), -1, False),
            ((1, 64, 9, 9), -1, False),
            ((1, 71, 7, 7), -1, False),
            ((1, 8, 1, 10), -1, False),
            ((1, 8, 1, 1), -1, False),
            ((1, 8, 1, 2), -1, False),
            ((1, 8, 1, 100 + 1), -1, False),
            ((1, 8, 10, 10), -1, False),
            ((1, 8, 2048, 256), -1, False),
            ((1, 8, 256, 2048), -1, False),
            ((1, 8, 256, 256), -1, False),
            ((1, 8, 300, 300), -1, False),
            ((12, 24, 24), -1, False),
            ((12, 50, 50), -1, False),
            ((16, 1, 60), -1, False),
            ((16, 1, 1000 + 1), -1, False),
            ((16, 19, 19), -1, False),
            ((16, 59, 59), -1, False),
            ((16, 6, 49, 49), -1, False),
            ((16, 6, 64, 64), -1, False),
            ((16, 7, 7), -1, False),
            ((16, 8, 49, 49), -1, False),
            ((16, 8, 64, 64), -1, False),
            ((4, 12, 49, 49), -1, False),
            ((4, 12, 64, 64), -1, False),
            ((4, 16, 49, 49), -1, False),
            ((4, 16, 64, 64), -1, False),
            ((64, 3, 49, 49), -1, False),
            ((64, 3, 64, 64), -1, False),
            ((64, 4, 49, 49), -1, False),
            ((64, 4, 64, 64), -1, False),
            ((8, 100, 100), -1, False),
            ((8, 100, 920), -1, False),
            ((8, 920, 920), -1, False),
        ],
    }
}


def run_softmax(device, params):
    [input_shape, dim, half_to_float] = params
    # TODO find out what half_to_float is supposed to mean in the provided traces
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.softmax(torch_input_tensor, dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.softmax(input_tensor, dim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.989
    tensors = [input_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", parameters["default"]["params"])
def test_softmax(device, params):
    run_softmax(device, params)


def run(
    params,
    *,
    device,
) -> list:
    return run_softmax(device, params)
