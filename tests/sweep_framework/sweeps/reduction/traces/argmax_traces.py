# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

TIMEOUT = 15

parameters = {
    "pytorch": {
        "height": [1, 2],
        "width": [7, 51865],
        "dim": [-1],
        "dtype": [ttnn.float32, ttnn.bfloat16],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    }
}


def run_argmax(device, height, width, dim, dtype, layout):
    input_tensor = torch.rand([height, width], dtype=torch.float32)
    ttnn_input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=layout, device=device)

    if dtype == ttnn.bfloat16:
        # Let torch input have same precision as the ttnn input
        torch_input_tensor = ttnn.to_torch(ttnn_input_tensor)
    else:
        torch_input_tensor = input_tensor

    torch_output_tensor = torch.argmax(torch_input_tensor, dim)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.argmax(ttnn_input_tensor, dim=dim)
    ttnn_output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [ttnn_input_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, ttnn_output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("height", parameters["pytorch"]["height"])
@pytest.mark.parametrize("width", parameters["pytorch"]["width"])
@pytest.mark.parametrize("dim", parameters["pytorch"]["dim"])
@pytest.mark.parametrize("dtype", parameters["pytorch"]["dtype"])
@pytest.mark.parametrize("layout", parameters["pytorch"]["layout"])
def test_pytorch(device, height, width, dim, dtype, layout):
    (result, msg), e2e_perf = run_argmax(device, height, width, dim, dtype, layout)
    assert result, msg
    logger.info(msg)
    if e2e_perf:
        logger.info(f"E2E Performance: {e2e_perf}")


def run(
    height,
    width,
    dim,
    dtype,
    layout,
    *,
    device,
) -> list:
    return run_argmax(device, height, width, dim, dtype, layout)
