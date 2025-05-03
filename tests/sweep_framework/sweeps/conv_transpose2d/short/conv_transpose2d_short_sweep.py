# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import os
import itertools
import random
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.conv_transpose2d_common import run_short, mesh_device_fixture

parameters = {
    "short_sweep_suite": {
        "input_specs": [
            # Contains following params
            # [batch_size, input_channels, input_height, input_width, output_channels, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, out_pad_h, out_pad_w]
            # [20, 16, 50, 100, 33, 3, 3, 2, 2, 0, 0, 1, 1, 0, 0], Batch size too big
            [1, 16, 50, 100, 33, 3, 3, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 1024, 14, 14, 512, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 128, 112, 112, 64, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 128, 64, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 16, 14, 14, 1, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 256, 32, 32, 128, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 256, 56, 56, 128, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 4, 7, 7, 16, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 512, 16, 16, 256, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            # [1, 512, 28, 28, 256, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
            [1, 64, 128, 128, 32, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0],
        ]
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    input_specs,
    *,
    device,
) -> list:
    return run_short(
        input_specs,
        device,
    )


import pytest


@pytest.mark.parametrize("input_spec", parameters["short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_transpose2d_localrun(device, input_spec):
    run_short(
        input_spec,
        device,
    )
