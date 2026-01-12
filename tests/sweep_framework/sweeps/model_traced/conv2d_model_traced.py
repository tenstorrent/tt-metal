# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import torch

from tests.sweep_framework.sweep_utils.conv2d_common import (
    run_conv2d_short_sweep,
    run_conv1d_short_sweep,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
# Conv2d operations can be slow, especially with large kernels/channels
TIMEOUT = 180

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("conv2d", all_cases=False)

parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_specs": [
            # Contains following params
            # [batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, groups, dilation_h, dilation_w, bias]
            # Use tuple so it serializes as a string for proper deserialization
            (1, 16, 8, 4, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1, False),
        ],
        "is_conv1d": [False],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_specs,
    is_conv1d=False,
    config_tensors_in_dram=False,
    *,
    device,
    **kwargs,
) -> list:
    # Call the short sweep function
    if is_conv1d:
        result = run_conv1d_short_sweep(input_specs, device)
    else:
        result = run_conv2d_short_sweep(input_specs, device, config_tensors_in_dram=config_tensors_in_dram)

    # Convert short_sweep format [pcc_bool, pcc_value, e2e_perf, output_tensor, expected_tensor]
    # to model_traced format [pcc_tuple, e2e_perf]
    # result[0]: bool (PCC passed/failed)
    # result[1]: float (actual PCC value)
    # result[2]: int/float (e2e performance time)

    pcc_passed = bool(result[0])
    pcc_value = float(result[1])
    e2e_perf = result[2]

    # Format as (bool, message) tuple expected by sweep framework
    pcc_result = (pcc_passed, f"PCC: {pcc_value:.6f}")

    return [pcc_result, e2e_perf]
