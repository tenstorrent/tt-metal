# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.sweep_framework.sweep_utils.conv2d_common import (
    run_conv2d_short_sweep,
    run_conv1d_short_sweep,
)
import ttnn

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
    compute_config=None,
    dtype=None,
    config_tensors_in_dram=False,
    *,
    device,
    **kwargs,
) -> list:
    # Parse compute_kernel_config from dict to ttnn object
    parsed_compute_config = None
    if compute_config and isinstance(compute_config, dict):
        math_fidelity_str = compute_config.get("math_fidelity", "HiFi4")
        math_fidelity_map = {
            "HiFi4": ttnn.MathFidelity.HiFi4,
            "HiFi3": ttnn.MathFidelity.HiFi3,
            "HiFi2": ttnn.MathFidelity.HiFi2,
            "LoFi": ttnn.MathFidelity.LoFi,
        }
        math_fidelity = math_fidelity_map.get(math_fidelity_str, ttnn.MathFidelity.HiFi4)
        fp32_dest_acc_en = bool(compute_config.get("fp32_dest_acc_en", 0))
        packer_l1_acc = bool(compute_config.get("packer_l1_acc", 0))

        parsed_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    # Parse output_dtype from string to ttnn dtype
    parsed_dtype = None
    if dtype and isinstance(dtype, str):
        dtype_map = {
            "bfloat16": ttnn.bfloat16,
            "bfloat8_b": ttnn.bfloat8_b,
            "float32": ttnn.float32,
            "uint16": ttnn.uint16,
            "uint32": ttnn.uint32,
            "int32": ttnn.int32,
        }
        parsed_dtype = dtype_map.get(dtype, ttnn.bfloat16)

    # Call the short sweep function with parsed ttnn objects
    if is_conv1d:
        result = run_conv1d_short_sweep(input_specs, device)
    else:
        result = run_conv2d_short_sweep(
            input_specs,
            device,
            config_tensors_in_dram=config_tensors_in_dram,
            output_dtype=parsed_dtype,
            compute_config=parsed_compute_config,
        )

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
