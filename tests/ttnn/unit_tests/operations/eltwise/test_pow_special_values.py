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
    0.0,
]

# Names for special values (for CSV readability)
SPECIAL_VALUE_NAMES = {
    float("inf"): "inf",
    float("-inf"): "-inf",
    -0.0: "minus 0.0",
    1.0: "1.0",
    0.0: "0.0",
}


def get_value_name(val):
    """Get readable name for a special value."""
    if math.isnan(val):
        return "nan"
    return SPECIAL_VALUE_NAMES.get(val, str(val))


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


def test_binary_pow_special_values(device):
    """
    Test binary pow with all combinations of special values.
    Tests: bf16, bf16-improved, fp32, fp32-improved
    Each with check_fractional_ulp True and False.

    Output CSV columns:
    - case: test configuration (e.g., bf16-false, fp32-improved-true)
    - input_a: input A value
    - input_b: input B value
    - ttnn_result: TTNN computed result
    - golden_result: Torch golden reference result
    - error: exception message if operation failed
    """
    csv_file_path = "binary_pow_special_values.csv"

    # Generate all combinations of special values for A and B
    combinations = list(itertools.product(SPECIAL_VALUES, SPECIAL_VALUES))

    print(f"Testing binary pow with {len(combinations)} special value combinations...")
    print(f"Special values: {[get_value_name(v) for v in SPECIAL_VALUES]}")

    # Test configurations
    test_configs = [
        ("bf16-false", ttnn.bfloat16, torch.bfloat16, False, False),
        ("bf16-true", ttnn.bfloat16, torch.bfloat16, True, False),
        ("bf16-improved-false", ttnn.bfloat16, torch.bfloat16, False, True),
        ("bf16-improved-true", ttnn.bfloat16, torch.bfloat16, True, True),
        ("fp32-false", ttnn.float32, torch.float32, False, False),
        ("fp32-true", ttnn.float32, torch.float32, True, False),
        ("fp32-improved-false", ttnn.float32, torch.float32, False, True),
        ("fp32-improved-true", ttnn.float32, torch.float32, True, True),
    ]

    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["case", "input_a", "input_b", "ttnn_result", "golden_result", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_cases = 0
        for case_name, ttnn_dtype, torch_dtype, check_fractional_ulp, use_improved in test_configs:
            print(f"Testing case: {case_name}")

            for a_val, b_val in combinations:
                a_name = get_value_name(a_val)
                b_name = get_value_name(b_val)

                # Create tensors with single value (tile-sized for TTNN)
                tensor_a = torch.full((1, 1, 32, 32), a_val, dtype=torch_dtype)
                tensor_b = torch.full((1, 1, 32, 32), b_val, dtype=torch_dtype)

                # Calculate golden result
                if check_fractional_ulp:
                    if torch_dtype == torch.bfloat16:
                        golden = torch.pow(tensor_a.to(torch.float32), tensor_b.to(torch.float32))
                    else:
                        golden = torch.pow(tensor_a.to(torch.float64), tensor_b.to(torch.float64))
                else:
                    golden = torch.pow(tensor_a, tensor_b)

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

                    if use_improved:
                        # Improved version using exp(b * log(a))
                        tt_result = ttnn.multiply(
                            tt_a,
                            tt_b,
                            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
                            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
                            use_legacy=False,
                        )
                    else:
                        # Standard binary pow
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


def test_unary_pow_special_values(device):
    """
    Test unary pow (scalar exponent) with all combinations of special values.
    Tests: bf16, bf16-improved, fp32, fp32-improved
    Each with check_fractional_ulp True and False.

    Note: Unary pow with negative exponents is not supported and will raise an error.
    This is documented in binary_composite_op.cpp:742: "works for positive exponents only"

    Output CSV columns:
    - case: test configuration (e.g., bf16-false, fp32-improved-true)
    - input_a: input A tensor value
    - input_b_scalar: input B scalar exponent value
    - ttnn_result: TTNN computed result
    - golden_result: Torch golden reference result
    - error: exception message if operation failed (e.g., "UNSUPPORTED: works for positive exponents only")
    """
    csv_file_path = "unary_pow_special_values.csv"

    # Generate all combinations of special values for A (tensor) and B (scalar)
    combinations = list(itertools.product(SPECIAL_VALUES, SPECIAL_VALUES))

    print(f"Testing unary pow with {len(combinations)} special value combinations...")
    print(f"Special values: {[get_value_name(v) for v in SPECIAL_VALUES]}")
    print("Note: Negative exponents are not supported for unary pow")

    # Test configurations
    test_configs = [
        ("bf16-false", ttnn.bfloat16, torch.bfloat16, False, False),
        ("bf16-true", ttnn.bfloat16, torch.bfloat16, True, False),
        ("bf16-improved-false", ttnn.bfloat16, torch.bfloat16, False, True),
        ("bf16-improved-true", ttnn.bfloat16, torch.bfloat16, True, True),
        ("fp32-false", ttnn.float32, torch.float32, False, False),
        ("fp32-true", ttnn.float32, torch.float32, True, False),
        ("fp32-improved-false", ttnn.float32, torch.float32, False, True),
        ("fp32-improved-true", ttnn.float32, torch.float32, True, True),
    ]

    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["case", "input_a", "input_b_scalar", "ttnn_result", "golden_result", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_cases = 0
        for case_name, ttnn_dtype, torch_dtype, check_fractional_ulp, use_improved in test_configs:
            print(f"Testing case: {case_name}")

            for a_val, b_scalar in combinations:
                a_name = get_value_name(a_val)
                b_name = get_value_name(b_scalar)

                # Create tensor with single value (tile-sized for TTNN)
                tensor_a = torch.full((1, 1, 32, 32), a_val, dtype=torch_dtype)

                # Calculate golden result
                if check_fractional_ulp:
                    if torch_dtype == torch.bfloat16:
                        golden = torch.pow(tensor_a.to(torch.float32), b_scalar)
                    else:
                        golden = torch.pow(tensor_a.to(torch.float64), b_scalar)
                else:
                    golden = torch.pow(tensor_a, b_scalar)

                golden_val = golden[0, 0, 0, 0].item()

                try:
                    # Convert to TTNN tensor
                    tt_a = ttnn.from_torch(
                        tensor_a,
                        dtype=ttnn_dtype,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                    if use_improved:
                        # Improved version using exp(b * log(a))
                        tt_result = ttnn.multiply(
                            tt_a,
                            b_scalar,
                            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
                            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
                            use_legacy=False,
                        )
                    else:
                        # Standard unary pow with scalar exponent
                        # Note: This will raise TT_FATAL for negative exponents
                        # Error message: "works for positive exponents only"
                        tt_result = ttnn.pow(tt_a, b_scalar)

                    result = ttnn.to_torch(tt_result)
                    ttnn_val = result[0, 0, 0, 0].item()
                    error_msg = ""

                except Exception as e:
                    ttnn_val = "ERROR"
                    error_msg = str(e)
                    # Check if it's the negative exponent error
                    if "positive exponents only" in str(e):
                        error_msg = "UNSUPPORTED: works for positive exponents only"

                writer.writerow(
                    {
                        "case": case_name,
                        "input_a": a_name,
                        "input_b_scalar": b_name,
                        "ttnn_result": format_result(ttnn_val) if ttnn_val != "ERROR" else "ERROR",
                        "golden_result": format_result(golden_val),
                        "error": error_msg,
                    }
                )
                total_cases += 1

    print(f"\nResults saved to {csv_file_path}")
    print(f"Total test cases: {total_cases}")
