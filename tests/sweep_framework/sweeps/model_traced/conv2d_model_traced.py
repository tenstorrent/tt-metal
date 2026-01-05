# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
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


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Invalidate conv2d test vectors that might cause timeouts or excessive memory usage.
    """
    input_specs = test_vector.get("input_specs")

    if input_specs is None:
        return False, None

    # Parse input specs - handle both tuple and string formats
    if isinstance(input_specs, str):
        try:
            input_specs = eval(input_specs)
        except:
            return False, None

    # input_specs format:
    # [batch_size, output_channels, input_channels, input_height, input_width,
    #  kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, groups, dilation_h, dilation_w, has_bias]
    if len(input_specs) < 15:
        return False, None

    batch_size = input_specs[0]
    output_channels = input_specs[1]
    input_channels = input_specs[2]
    input_height = input_specs[3]
    input_width = input_specs[4]
    kernel_height = input_specs[5]
    kernel_width = input_specs[6]

    # Calculate total tensor sizes to detect memory-intensive configs
    input_size = batch_size * input_channels * input_height * input_width
    weight_size = output_channels * input_channels * kernel_height * kernel_width
    output_size = batch_size * output_channels * input_height * input_width  # Approximate

    total_elements = input_size + weight_size + output_size

    # Skip extremely large configurations that are likely to timeout
    # These thresholds are empirical - adjust based on observed failures
    if total_elements > 50_000_000:  # 50M elements
        return True, f"conv2d: Configuration too large ({total_elements:,} total elements) - likely to timeout"

    # Skip configurations with very large kernels or feature maps that might be unstable
    if input_height > 1024 or input_width > 1024:
        return True, f"conv2d: Input dimensions too large ({input_height}x{input_width})"

    if kernel_height > 32 or kernel_width > 32:
        return True, f"conv2d: Kernel too large ({kernel_height}x{kernel_width})"

    return False, None


def run(
    input_specs,
    is_conv1d=False,
    compute_config=None,
    dtype=None,
    config_tensors_in_dram=False,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, config_id, etc.
) -> list:
    # Call the short sweep function
    if is_conv1d:
        result = run_conv1d_short_sweep(input_specs, device)
    else:
        result = run_conv2d_short_sweep(input_specs, device, config_tensors_in_dram=config_tensors_in_dram)

    # Convert short_sweep format [pcc_bool, perf, timestamp, tensor1, tensor2]
    # to model_traced format [pcc_tuple, e2e_perf]
    pcc_passed = result[0]
    e2e_perf = result[1]
    pcc_message = f"PCC: {e2e_perf:.6f}" if pcc_passed else "PCC check failed"

    return [(pcc_passed, pcc_message), e2e_perf]
