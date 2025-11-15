# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import os
import itertools
import random
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.conv2d_common import (
    run_conv2d_short_sweep,
    run_conv1d_short_sweep,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 60

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("conv2d", all_cases=False)

parameters = {
    "short_sweep_suite_conv2d": {
        "input_specs": [
            # Contains following params
            # [batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, groups, dilation_h, dilation_w, bias]
            [1, 16, 8, 4, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1, False],
        ],
        "is_conv1d": [False],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    input_specs,
    is_conv1d=False,
    *,
    device,
) -> list:
    # Call the short sweep function
    if is_conv1d:
        result = run_conv1d_short_sweep(input_specs, device)
    else:
        result = run_conv2d_short_sweep(input_specs, device)

    # Convert short_sweep format [pcc_bool, perf, timestamp, tensor1, tensor2]
    # to model_traced format [pcc_tuple, e2e_perf]
    pcc_passed = result[0]
    e2e_perf = result[1]
    pcc_message = f"PCC: {e2e_perf:.6f}" if pcc_passed else "PCC check failed"

    return [(pcc_passed, pcc_message), e2e_perf]
