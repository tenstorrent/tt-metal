# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import csv
import math
import itertools
from datetime import datetime


# Special values to test
SPECIAL_VALUES = [
    float("inf"),
    float("nan"),
    float("-inf"),
    -0.0,
    1.0,
    -1.0,
    0.0,
]


def get_value_name(val):
    """Get readable name for a special value."""
    if math.isnan(val):
        return "nan"
    elif math.isinf(val):
        return "inf" if val > 0 else "-inf"
    elif val == 0.0:
        # Check for negative zero using copysign
        # Note: -0.0 == 0.0 is True in Python, so we need copysign to distinguish
        if math.copysign(1.0, val) < 0:
            return "minus 0.0"
        return "0.0"
    elif val == 1.0:
        return "1.0"
    return str(val)


def format_result(val):
    """Format result value for CSV output."""
    if isinstance(val, torch.Tensor):
        val = val.item()
    if math.isnan(val):
        return "nan"
    elif math.isinf(val):
        return "inf" if val > 0 else "-inf"
    elif val == 0.0:
        # Check for negative zero
        if math.copysign(1.0, val) < 0:
            return "minus 0.0"
        return "0.0"
    return str(val)


def test_fmod_special_values(device):
    """
    Test fmod with all combinations of special values.
    Compares torch.fmod vs ttnn.fmod for bf16 and fp32.

    Output CSV columns:
    - case: test configuration (e.g., bf16, fp32)
    - input_a: input A value (dividend)
    - input_b: input B value (divisor)
    - ttnn_result: TTNN computed result
    - golden_result: Torch golden reference result
    - error: exception message if operation failed
    """
    csv_file_path = "fmod_special_values.csv"

    # Generate all combinations of special values for A and B
    combinations = list(itertools.product(SPECIAL_VALUES, SPECIAL_VALUES))

    print(f"Testing fmod with {len(combinations)} special value combinations...")
    print(f"Special values: {[get_value_name(v) for v in SPECIAL_VALUES]}")

    # Test configurations: (name, ttnn_dtype, torch_dtype)
    test_configs = [
        ("bf16", ttnn.bfloat16, torch.bfloat16),
        ("fp32", ttnn.float32, torch.float32),
    ]

    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["case", "input_a", "input_b", "ttnn_result", "golden_result", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_cases = 0
        for case_name, ttnn_dtype, torch_dtype in test_configs:
            print(f"Testing case: {case_name}")

            for a_val, b_val in combinations:
                a_name = get_value_name(a_val)
                b_name = get_value_name(b_val)

                # Create tensors with single value (tile-sized for TTNN)
                tensor_a = torch.full((1, 1, 32, 32), a_val, dtype=torch_dtype)
                tensor_b = torch.full((1, 1, 32, 32), b_val, dtype=torch_dtype)

                # Calculate golden result using torch.fmod
                golden = torch.fmod(tensor_a, tensor_b)
                golden_val = golden[0, 0, 0, 0].item()

                try:
                    # Convert to TTNN tensors
                    tt_a = ttnn.from_torch(
                        tensor_a,
                        dtype=ttnn_dtype,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    tt_b = ttnn.from_torch(
                        tensor_b,
                        dtype=ttnn_dtype,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                    # Compute fmod using TTNN
                    tt_result = ttnn.pow(tt_a, tt_b)

                    result = ttnn.to_torch(tt_result)
                    ttnn_val = result[0, 0, 0, 0].item()
                    error_msg = ""

                except Exception as e:
                    ttnn_val = "ERROR"
                    error_msg = str(e)

                writer.writerow(
                    {
                        "case": case_name,
                        "input_a": a_name,
                        "input_b": b_name,
                        "ttnn_result": format_result(ttnn_val) if ttnn_val != "ERROR" else "ERROR",
                        "golden_result": format_result(golden_val),
                        "error": error_msg,
                    }
                )
                total_cases += 1

    print(f"\nResults saved to {csv_file_path}")
    print(f"Total test cases: {total_cases}")
