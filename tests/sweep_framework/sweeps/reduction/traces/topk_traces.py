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
    "pytorch": {
        "params": [
            ((1, 5), 3),
            ((1, 32), 3),
            ((1, 50), 50),
            ((1, 50), 50257),
        ],
    }
}


def run_topk(device, params):
    [input_shape, k] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.topk(torch_input_tensor, k)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.topk(input_tensor, k)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", parameters["pytorch"]["params"])
def test_pytorch(device, params):
    run_topk(device, params)


def run(
    params,
    *,
    device,
) -> list:
    return run_topk(device, params)
