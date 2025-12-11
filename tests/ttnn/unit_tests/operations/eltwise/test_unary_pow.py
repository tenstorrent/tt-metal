# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import csv
import math
from datetime import datetime
from tests.ttnn.utils_for_testing import assert_with_ulp
from models.common.utility_functions import comp_ulp_check


def _generate_summary_report(results, summary_file_path, test_name, data_type, num_a_values, num_b_values):
    """
    Generate a detailed summary report for binary power ULP analysis.

    Args:
        results: List of tuples (input_b, max_ulp)
        summary_file_path: Path to save the summary report
        test_name: Name of the test (e.g., "Binary Power (BF16)")
        data_type: Data type used (e.g., "BFloat16", "Float32")
        num_a_values: Number of A values tested
        num_b_values: Number of B values tested
    """
    if not results:
        return

    # Separate results into three categories: finite, infinite, and NaN
    finite_results = [(b, ulp) for b, ulp in results if math.isfinite(ulp)]
    inf_results = [(b, ulp) for b, ulp in results if math.isinf(ulp)]
    nan_results = [(b, ulp) for b, ulp in results if math.isnan(ulp)]

    total_tests = len(results)
    num_finite = len(finite_results)
    num_inf = len(inf_results)
    num_nan = len(nan_results)

    # Calculate statistics for finite values only
    if finite_results:
        finite_ulp_values = [ulp for _, ulp in finite_results]
        max_ulp_finite = max(finite_ulp_values)
        min_ulp_finite = min(finite_ulp_values)

        # Count values in different ULP ranges (finite only)
        ulp_0 = sum(1 for ulp in finite_ulp_values if ulp == 0)
        ulp_0_to_1 = sum(1 for ulp in finite_ulp_values if 0 < ulp <= 1)
        ulp_1_to_2 = sum(1 for ulp in finite_ulp_values if 1 < ulp <= 2)
        ulp_2_to_5 = sum(1 for ulp in finite_ulp_values if 2 < ulp <= 5)
        ulp_5_to_10 = sum(1 for ulp in finite_ulp_values if 5 < ulp <= 10)
        ulp_10_to_100 = sum(1 for ulp in finite_ulp_values if 10 < ulp <= 100)
        ulp_above_100 = sum(1 for ulp in finite_ulp_values if ulp > 100)

        # Find worst cases (top 10 highest finite ULP values)
        sorted_finite_results = sorted(finite_results, key=lambda x: x[1], reverse=True)
        worst_cases = sorted_finite_results[:10]
    else:
        max_ulp_finite = min_ulp_finite = 0
        ulp_0 = ulp_0_to_1 = ulp_1_to_2 = ulp_2_to_5 = ulp_5_to_10 = ulp_10_to_100 = ulp_above_100 = 0
        worst_cases = []

    # Calculate pass rate (ULP <= 1 is typically considered passing, finite only)
    pass_count_finite = ulp_0 + ulp_0_to_1
    pass_rate_finite = (pass_count_finite / num_finite * 100) if num_finite > 0 else 0
    # Overall pass rate considers inf and nan as failures
    pass_rate_overall = (pass_count_finite / total_tests * 100) if total_tests > 0 else 0

    # Generate report
    with open(summary_file_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"         BINARY POWER OPERATION - ULP ACCURACY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Name: {test_name}\n")
        f.write(f"Data Type: {data_type}\n\n")

        f.write("-" * 80 + "\n")
        f.write("                           TEST CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Input A Tensor Size:     {num_a_values:,} values\n")
        f.write(f"  Input B Values Tested:   {num_b_values:,} values\n")
        f.write(f"  Total Comparisons:       {total_tests:,}\n")
        f.write(f"  Operation:               A^B (element-wise power)\n\n")

        f.write("-" * 80 + "\n")
        f.write("                        ULP CATEGORY BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Finite ULP Cases:        {num_finite:>10,} ({(num_finite/total_tests)*100:>6.2f}%)\n")
        f.write(f"  Infinite ULP Cases:      {num_inf:>10,} ({(num_inf/total_tests)*100:>6.2f}%)\n")
        f.write(f"  NaN ULP Cases:           {num_nan:>10,} ({(num_nan/total_tests)*100:>6.2f}%)\n")
        f.write(f"  {'─'*50}\n")
        f.write(f"  Total:                   {total_tests:>10,} (100.00%)\n\n")

        if num_finite > 0:
            f.write("-" * 80 + "\n")
            f.write("                    FINITE ULP STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Maximum ULP Error:       {max_ulp_finite:.6f}\n")
            f.write(f"  Minimum ULP Error:       {min_ulp_finite:.6f}\n")
            f.write(f"  Percentage of ULP <= 1 : \n")
            f.write(f"  - Finite cases only:     {pass_rate_finite:.2f}%\n")
            f.write(f"  - Incl non-finite cases: {pass_rate_overall:.2f}%\n\n")

            f.write("-" * 80 + "\n")
            f.write("                    FINITE ULP DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"  ULP = 0 (Exact Match):   {ulp_0:>10,} ({(ulp_0/num_finite)*100:>6.2f}%)\n")
            f.write(f"  0 < ULP <= 1:            {ulp_0_to_1:>10,} ({(ulp_0_to_1/num_finite)*100:>6.2f}%)\n")
            f.write(f"  1 < ULP <= 2:            {ulp_1_to_2:>10,} ({(ulp_1_to_2/num_finite)*100:>6.2f}%)\n")
            f.write(f"  2 < ULP <= 5:            {ulp_2_to_5:>10,} ({(ulp_2_to_5/num_finite)*100:>6.2f}%)\n")
            f.write(f"  5 < ULP <= 10:           {ulp_5_to_10:>10,} ({(ulp_5_to_10/num_finite)*100:>6.2f}%)\n")
            f.write(f"  10 < ULP <= 100:         {ulp_10_to_100:>10,} ({(ulp_10_to_100/num_finite)*100:>6.2f}%)\n")
            f.write(f"  ULP > 100:               {ulp_above_100:>10,} ({(ulp_above_100/num_finite)*100:>6.2f}%)\n\n")

        if num_inf > 0:
            f.write("-" * 80 + "\n")
            f.write("                    INFINITE ULP CASES\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Infinite Cases:    {num_inf:,}\n\n")
            # Format as comma-separated list on one line
            sample_b_values = [f"{b_val:.10g}" for b_val, _ in inf_results[:20]]
            b_list_str = " , ".join(sample_b_values)
            if len(inf_results) > 20:
                f.write(f"  For Input b = {b_list_str} and {len(inf_results) - 20} more\n\n")
            else:
                f.write(f"  For Input b = {b_list_str}\n\n")

        if num_nan > 0:
            f.write("-" * 80 + "\n")
            f.write("                    NaN ULP CASES\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total NaN Cases:         {num_nan:,}\n\n")
            # Format as comma-separated list on one line
            sample_b_values = [f"{b_val:.10g}" for b_val, _ in nan_results[:20]]
            b_list_str = " , ".join(sample_b_values)
            if len(nan_results) > 20:
                f.write(f"  For Input b = {b_list_str} and {len(nan_results) - 20} more\n\n")
            else:
                f.write(f"  For Input b = {b_list_str}\n\n")

        if worst_cases:
            f.write("-" * 80 + "\n")
            f.write("                    WORST FINITE CASES (Highest ULP)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {'Rank':<6} {'Input B':<25} {'Max ULP':<20}\n")
            f.write(f"  {'-'*6} {'-'*25} {'-'*20}\n")
            for i, (b_val, ulp_val) in enumerate(worst_cases, 1):
                f.write(f"  {i:<6} {b_val:<25.10g} {ulp_val:<20.6f}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("                              CONCLUSION\n")
        f.write("-" * 80 + "\n")

        # Determine status based on overall metrics
        has_issues = num_inf > 0 or num_nan > 0
        if pass_rate_overall >= 99.0 and not has_issues:
            status = "EXCELLENT"
            description = "The binary power operation demonstrates excellent numerical accuracy."
        elif pass_rate_overall >= 95.0 and num_inf + num_nan < total_tests * 0.01:
            status = "GOOD"
            description = "The binary power operation shows good numerical accuracy with minor deviations."
        elif pass_rate_overall >= 90.0:
            status = "ACCEPTABLE"
            description = "The binary power operation shows acceptable accuracy but some edge cases may need attention."
        else:
            status = "NEEDS IMPROVEMENT"
            description = (
                "The binary power operation shows significant numerical errors that may require investigation."
            )

        f.write(f"\n  Overall Status: {status}\n\n")
        f.write(f"  {description}\n\n")
        f.write(f"  Key Findings:\n")
        f.write(f"    - {pass_count_finite:,} out of {total_tests:,} test cases passed (ULP <= 1)\n")
        if num_finite > 0:
            f.write(f"    - Maximum observed finite ULP error: {max_ulp_finite:.6f}\n")
        if num_inf > 0:
            f.write(f"    - {num_inf:,} cases produced Infinite ULP (possible overflow or undefined result)\n")
        if num_nan > 0:
            f.write(f"    - {num_nan:,} cases produced NaN ULP (possible invalid computation)\n")
        if ulp_above_100 > 0:
            f.write(f"    - {ulp_above_100:,} cases with finite ULP > 100 may indicate edge case issues\n")
        if ulp_0 > 0:
            f.write(f"    - {ulp_0:,} cases achieved exact match (ULP = 0)\n")

        if num_inf > 0 or num_nan > 0:
            f.write(f"\n  Recommendations:\n")
            if num_inf > 0:
                f.write(f"    - Review {num_inf:,} Infinite ULP cases for potential overflow conditions\n")
            if num_nan > 0:
                f.write(f"    - Investigate {num_nan:,} NaN ULP cases for invalid input combinations\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("                           END OF REPORT\n")
        f.write("=" * 80 + "\n")


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0, 3.0])
def test_pow_arange_masking_unary(exponent, device):
    # Generate all possible bit pattern for bf16
    tt_input = generate_clean_bf16_tensor()

    tt_in = ttnn.from_torch(
        tt_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(tt_input, exponent, device=device)

    tt_result = ttnn.pow(tt_in, exponent)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


def test_pow_arange_masking_binary(device):
    # Generate all possible bit pattern for bf16
    tt_input = generate_clean_bf16_tensor()

    tt_in = ttnn.from_torch(
        tt_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden = torch.pow(tt_input, tt_input)

    tt_result = ttnn.pow(tt_in, tt_in)
    result = ttnn.to_torch(tt_result)

    # Run comp_ulp_check to get max ULP
    max_ulp = comp_ulp_check(
        golden=golden,
        calculated=result,
        allow_nonfinite=True,
    )
    print(f"Max ULP: {max_ulp}")

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


def generate_clean_bf16_tensor(dtype=torch.bfloat16):
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)  # 65536 values
    fp32 = input_tensor.to(torch.float32)

    # Remove special values (NaN, -0.0, +inf, -inf, subnormals)
    neg_zero_mask = (fp32 == 0.0) & torch.signbit(fp32)
    tiny = torch.finfo(torch.bfloat16).tiny  # 2**-126
    good_mask = torch.isfinite(fp32) & ~neg_zero_mask & (fp32.abs() >= tiny)
    fp32 = fp32[good_mask]  # 65024 values

    # Filter bf16 values to [±1e-15, ±1e15]
    abs_fp32 = fp32.abs()
    mask = (abs_fp32 >= 1e-15) & (abs_fp32 <= 1e15)
    fp32 = fp32[mask]  # 25510 values

    return fp32.to(dtype)


@pytest.mark.parametrize("check_fractional_ulp", [True, False])
def test_binary_pow_sweep_BF16_test(device, check_fractional_ulp):
    # Generate clean bf16 tensor (tensor A)
    tensor_a = generate_clean_bf16_tensor()
    num_values = tensor_a.numel()

    # CSV and summary file paths based on check_fractional_ulp parameter
    fractional_suffix = "true" if check_fractional_ulp else "false"
    csv_file_path = f"binary_pow_bf16_results_{fractional_suffix}.csv"
    summary_file_path = f"binary_pow_bf16_summary_{fractional_suffix}.txt"

    print(f"Testing binary pow (BF16) with {num_values} B values...")
    print(f"Each iteration tests {num_values} element-wise A^B operations")

    # Collect results for summary
    all_results = []

    # Open CSV file for writing results
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input_B", "Max_ULP"])

        # Iterate through each value in tensor_a as the B value
        for i in range(num_values):
            # Get the i-th value from tensor_a to use as B
            b_val = tensor_a[i]
            b_scalar = b_val.item()

            # Create tensor B filled with this single value
            tensor_b = torch.full_like(tensor_a, b_scalar, dtype=torch.bfloat16)

            # Convert to ttnn tensors
            tt_a = ttnn.from_torch(
                tensor_a,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_b = ttnn.from_torch(
                tensor_b,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Calculate golden result using torch
            if not check_fractional_ulp:
                golden = torch.pow(tensor_a, tensor_b)  # bf16 vs bf16
            else:
                golden = torch.pow(tensor_a.to(torch.float32), tensor_b.to(torch.float32))  # bf16 vs fp32

            # Run ttnn binary pow
            tt_result = ttnn.pow(tt_a, tt_b)
            result = ttnn.to_torch(tt_result)

            # Run comp_ulp_check to get max ULP
            max_ulp = comp_ulp_check(
                golden=golden,
                calculated=result,
                allow_nonfinite=True,
            )

            # Write to CSV
            csv_writer.writerow([b_scalar, max_ulp])
            all_results.append((b_scalar, max_ulp))

            # Print progress every 1000 iterations
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_values} B values...")

    # Generate summary report
    _generate_summary_report(
        results=all_results,
        summary_file_path=summary_file_path,
        test_name="Binary Power (BF16)",
        data_type="BFloat16",
        num_a_values=num_values,
        num_b_values=num_values,
    )

    print(f"\nResults saved to {csv_file_path}")
    print(f"Summary report saved to {summary_file_path}")


@pytest.mark.parametrize("check_fractional_ulp", [True, False])
def test_binary_pow_sweep_FP32_test(device, check_fractional_ulp):
    # Generate clean bf16 tensor (tensor A)
    tensor_a = generate_clean_bf16_tensor(torch.float32)
    num_values = tensor_a.numel()

    # CSV and summary file paths based on check_fractional_ulp parameter
    fractional_suffix = "true" if check_fractional_ulp else "false"
    csv_file_path = f"binary_pow_fp32_results_{fractional_suffix}.csv"
    summary_file_path = f"binary_pow_fp32_summary_{fractional_suffix}.txt"

    print(f"Testing binary pow (FP32) with {num_values} B values...")
    print(f"Each iteration tests {num_values} element-wise A^B operations")

    # Collect results for summary
    all_results = []

    # Open CSV file for writing results
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input_B", "Max_ULP"])

        # Iterate through each value in tensor_a as the B value
        for i in range(num_values):
            # Get the i-th value from tensor_a to use as B
            b_val = tensor_a[i]
            b_scalar = b_val.item()

            # Create tensor B filled with this single value
            tensor_b = torch.full_like(tensor_a, b_scalar, dtype=torch.float32)

            # Convert to ttnn tensors
            tt_a = ttnn.from_torch(
                tensor_a,
                dtype=ttnn.float32,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_b = ttnn.from_torch(
                tensor_b,
                dtype=ttnn.float32,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Calculate golden result using torch (fp64 for higher precision reference)
            if not check_fractional_ulp:
                golden = torch.pow(tensor_a, tensor_b)  # fp32 vs fp32
            else:
                golden = torch.pow(tensor_a.to(torch.float64), tensor_b.to(torch.float64))  # fp32 vs fp64

            # Run ttnn binary pow
            tt_result = ttnn.pow(tt_a, tt_b)
            result = ttnn.to_torch(tt_result)

            # Run comp_ulp_check to get max ULP
            max_ulp = comp_ulp_check(
                golden=golden,
                calculated=result,
                allow_nonfinite=True,
            )

            # Write to CSV
            csv_writer.writerow([b_scalar, max_ulp])
            all_results.append((b_scalar, max_ulp))

            # Print progress every 1000 iterations
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_values} B values...")

    # Generate summary report
    _generate_summary_report(
        results=all_results,
        summary_file_path=summary_file_path,
        test_name="Binary Power (FP32)",
        data_type="Float32",
        num_a_values=num_values,
        num_b_values=num_values,
    )

    print(f"\nResults saved to {csv_file_path}")
    print(f"Summary report saved to {summary_file_path}")


# **************************************************
# **************************************************

# Check with new exp, log
# Exp Improved for fp32
# Log improved for both dtypes

# **************************************************
# **************************************************


def test_pow_arange_masking_binary_improved(device):
    # Generate all possible bit pattern for bf16
    tt_input = generate_clean_bf16_tensor()

    tt_in = ttnn.from_torch(
        tt_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden = torch.pow(tt_input, tt_input)

    # tt_result = ttnn.pow(tt_in, tt_in)
    tt_result = ttnn.multiply(
        tt_in,
        tt_in,
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
        use_legacy=False,
    )
    result = ttnn.to_torch(tt_result)
    print(f"Result: {result}")

    # Run comp_ulp_check to get max ULP
    max_ulp = comp_ulp_check(
        golden=golden,
        calculated=result,
        allow_nonfinite=True,
    )
    print(f"Max ULP: {max_ulp}")

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


@pytest.mark.parametrize("check_fractional_ulp", [True, False])
def test_binary_pow_sweep_BF16_test_improved(device, check_fractional_ulp):
    # Generate clean bf16 tensor (tensor A)
    tensor_a = generate_clean_bf16_tensor()
    num_values = tensor_a.numel()

    # CSV and summary file paths based on check_fractional_ulp parameter
    fractional_suffix = "true" if check_fractional_ulp else "false"
    csv_file_path = f"binary_pow_bf16_improved_results_{fractional_suffix}.csv"
    summary_file_path = f"binary_pow_bf16_improved_summary_{fractional_suffix}.txt"

    print(f"Testing binary pow (BF16) with {num_values} B values...")
    print(f"Each iteration tests {num_values} element-wise A^B operations")

    # Collect results for summary
    all_results = []

    # Open CSV file for writing results
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input_B", "Max_ULP"])

        # Iterate through each value in tensor_a as the B value
        for i in range(num_values):
            # Get the i-th value from tensor_a to use as B
            b_val = tensor_a[i]
            b_scalar = b_val.item()

            # Create tensor B filled with this single value
            tensor_b = torch.full_like(tensor_a, b_scalar, dtype=torch.bfloat16)

            # Convert to ttnn tensors
            tt_a = ttnn.from_torch(
                tensor_a,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_b = ttnn.from_torch(
                tensor_b,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Calculate golden result using torch
            if not check_fractional_ulp:
                golden = torch.pow(tensor_a, tensor_b)  # bf16 vs bf16
            else:
                golden = torch.pow(tensor_a.to(torch.float32), tensor_b.to(torch.float32))  # bf16 vs fp32

            # Run ttnn binary pow
            # tt_result = ttnn.pow(tt_a, tt_b)
            tt_result = ttnn.multiply(
                tt_a,
                tt_b,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
                use_legacy=False,
            )
            result = ttnn.to_torch(tt_result)

            # Run comp_ulp_check to get max ULP
            max_ulp = comp_ulp_check(
                golden=golden,
                calculated=result,
                allow_nonfinite=True,
            )

            # Write to CSV
            csv_writer.writerow([b_scalar, max_ulp])
            all_results.append((b_scalar, max_ulp))

            # Print progress every 1000 iterations
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_values} B values...")

    # Generate summary report
    _generate_summary_report(
        results=all_results,
        summary_file_path=summary_file_path,
        test_name="Binary Power (BF16)",
        data_type="BFloat16",
        num_a_values=num_values,
        num_b_values=num_values,
    )

    print(f"\nResults saved to {csv_file_path}")
    print(f"Summary report saved to {summary_file_path}")


@pytest.mark.parametrize("check_fractional_ulp", [True, False])
def test_binary_pow_sweep_FP32_test_improved(device, check_fractional_ulp):
    # Generate clean bf16 tensor (tensor A)
    tensor_a = generate_clean_bf16_tensor(torch.float32)
    num_values = tensor_a.numel()

    # CSV and summary file paths based on check_fractional_ulp parameter
    fractional_suffix = "true" if check_fractional_ulp else "false"
    csv_file_path = f"binary_pow_fp32_improved_results_{fractional_suffix}.csv"
    summary_file_path = f"binary_pow_fp32_improved_summary_{fractional_suffix}.txt"

    print(f"Testing binary pow (FP32) with {num_values} B values...")
    print(f"Each iteration tests {num_values} element-wise A^B operations")

    # Collect results for summary
    all_results = []

    # Open CSV file for writing results
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input_B", "Max_ULP"])

        # Iterate through each value in tensor_a as the B value
        for i in range(num_values):
            # Get the i-th value from tensor_a to use as B
            b_val = tensor_a[i]
            b_scalar = b_val.item()

            # Create tensor B filled with this single value
            tensor_b = torch.full_like(tensor_a, b_scalar, dtype=torch.float32)

            # Convert to ttnn tensors
            tt_a = ttnn.from_torch(
                tensor_a,
                dtype=ttnn.float32,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_b = ttnn.from_torch(
                tensor_b,
                dtype=ttnn.float32,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Calculate golden result using torch (fp64 for higher precision reference)
            if not check_fractional_ulp:
                golden = torch.pow(tensor_a, tensor_b)  # fp32 vs fp32
            else:
                golden = torch.pow(tensor_a.to(torch.float64), tensor_b.to(torch.float64))  # fp32 vs fp64

            # Run ttnn binary pow
            # tt_result = ttnn.pow(tt_a, tt_b)
            tt_result = ttnn.multiply(
                tt_a,
                tt_b,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
                use_legacy=False,
            )
            result = ttnn.to_torch(tt_result)

            # Run comp_ulp_check to get max ULP
            max_ulp = comp_ulp_check(
                golden=golden,
                calculated=result,
                allow_nonfinite=True,
            )

            # Write to CSV
            csv_writer.writerow([b_scalar, max_ulp])
            all_results.append((b_scalar, max_ulp))

            # Print progress every 1000 iterations
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_values} B values...")

    # Generate summary report
    _generate_summary_report(
        results=all_results,
        summary_file_path=summary_file_path,
        test_name="Binary Power (FP32)",
        data_type="Float32",
        num_a_values=num_values,
        num_b_values=num_values,
    )

    print(f"\nResults saved to {csv_file_path}")
    print(f"Summary report saved to {summary_file_path}")
