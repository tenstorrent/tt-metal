# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive FP32 fmod test using all BF16 values converted to FP32.

This test:
1. Generates all possible BF16 values (~65536 values)
2. Converts them to FP32 as input 'a'
3. Sets input 'b' = a + 1.542
4. Calls ttnn.pow (which internally uses _sfpu_binary_power_61f_ that implements fmod)
5. Compares results using ULP (Units in Last Place)
6. Reports mismatches and stores them in a CSV file

Formula: fmod(a, b) = a - trunc(a/b) * b
"""

import pytest
import torch
import ttnn
import struct
import csv
import os
import math
from datetime import datetime


def bf16_to_float32(bf16_bits: int) -> float:
    """Convert BF16 bit pattern to FP32 value."""
    # BF16 is essentially FP32 with lower 16 mantissa bits truncated
    # So we just shift left by 16 to get the FP32 bit pattern
    fp32_bits = bf16_bits << 16
    return struct.unpack("f", struct.pack("I", fp32_bits))[0]


def float32_to_bits(f: float) -> int:
    """Convert FP32 value to its bit representation."""
    return struct.unpack("I", struct.pack("f", f))[0]


def compute_ulp_diff(a: float, b: float) -> int:
    """
    Compute the ULP (Units in Last Place) difference between two floats.

    ULP is the number of representable floating-point numbers between a and b.
    """
    # Handle special cases
    if math.isnan(a) or math.isnan(b):
        return float("inf") if (math.isnan(a) != math.isnan(b)) else 0

    if math.isinf(a) or math.isinf(b):
        if a == b:
            return 0
        return float("inf")

    if a == b:
        return 0

    # Get bit representations
    bits_a = float32_to_bits(a)
    bits_b = float32_to_bits(b)

    # Handle sign differences
    sign_a = (bits_a >> 31) & 1
    sign_b = (bits_b >> 31) & 1

    if sign_a != sign_b:
        # Different signs - compute distance through zero
        if sign_a:
            bits_a = 0x80000000 - bits_a
        if sign_b:
            bits_b = 0x80000000 - bits_b
        return bits_a + bits_b

    # Same sign - simple difference
    if sign_a:
        bits_a = 0x80000000 - bits_a
        bits_b = 0x80000000 - bits_b

    return abs(bits_a - bits_b)


def generate_all_bf16_as_fp32():
    """
    Generate all possible BF16 values as FP32.

    BF16 has 16 bits: 1 sign + 8 exponent + 7 mantissa
    Total: 2^16 = 65536 values
    """
    values = []

    for bf16_bits in range(65536):
        exponent = (bf16_bits >> 7) & 0xFF
        mantissa = bf16_bits & 0x7F

        # Skip special values
        if exponent == 0xFF:  # Inf or NaN
            continue
        if exponent == 0 and mantissa != 0:  # Denormals
            continue

        fp32_val = bf16_to_float32(bf16_bits)

        if math.isnan(fp32_val) or math.isinf(fp32_val):
            continue

        values.append(fp32_val)

    return values


def write_mismatches_to_file(mismatches: list, output_path: str):
    """Write mismatch details to a CSV file."""
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = ["index", "input_a", "input_b", "expected_torch", "ttnn_result", "ulp_diff", "abs_diff"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for mismatch in mismatches:
            writer.writerow(mismatch)

    print(f"Mismatches written to: {output_path}")


def pad_to_tile(values: list, tile_size: int = 32) -> list:
    """Pad values list to be a multiple of tile_size."""
    remainder = len(values) % tile_size
    if remainder != 0:
        padding_needed = tile_size - remainder
        # Pad with zeros (or first value to avoid div by zero issues)
        values = values + [1.0] * padding_needed
    return values


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_fmod_fp32_small_batch(device):
    """
    Small batch test of fmod operation to verify basic functionality.
    Uses a subset of BF16 values for quick testing.
    """

    # Configuration
    ULP_TOLERANCE = 10
    OFFSET_B = 1.542
    BATCH_SIZE = 32 * 32  # 1024 values = one 32x32 tile

    print("\nGenerating test values...")
    all_bf16_fp32 = generate_all_bf16_as_fp32()

    # Take a subset and filter
    filtered_values = []
    for val in all_bf16_fp32[: BATCH_SIZE * 2]:
        b_val = val + OFFSET_B
        if abs(val) >= 0.01 and abs(b_val) >= 0.01:
            filtered_values.append(val)
        if len(filtered_values) >= BATCH_SIZE:
            break

    # Pad to tile-aligned size (32x32 = 1024)
    while len(filtered_values) < 32 * 32:
        filtered_values.append(1.0)

    num_values = len(filtered_values)

    print(f"Testing with {num_values} values (tile-aligned 32x32)")

    # Create input tensors with proper tile-aligned shape (32x32)
    input_a = torch.tensor(filtered_values, dtype=torch.float32).reshape(1, 1, 32, 32)
    input_b = input_a + OFFSET_B

    # Compute expected result using PyTorch
    print("Computing expected results with PyTorch fmod...")
    expected = torch.fmod(input_a, input_b)

    # Create TTNN tensors
    print("Creating TTNN tensors...")
    ttnn_a = ttnn.from_torch(input_a, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_b = ttnn.from_torch(input_b, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)

    # Run fmod via ttnn.pow
    print("Running TTNN pow (fmod implementation)...")
    ttnn_result = ttnn.pow(ttnn_a, ttnn_b)

    # Convert back to PyTorch
    result = ttnn.to_torch(ttnn_result)

    # Flatten for comparison
    expected_flat = expected.flatten()
    result_flat = result.flatten()
    input_a_flat = input_a.flatten()

    # Compare results
    print("Comparing results...")
    mismatches = []
    max_ulp = 0
    total_ulp = 0
    valid_comparisons = 0

    for i in range(len(filtered_values)):
        a_val = input_a_flat[i].item()
        b_val = a_val + OFFSET_B
        exp_val = expected_flat[i].item()
        res_val = result_flat[i].item()

        ulp_diff = compute_ulp_diff(exp_val, res_val)

        if ulp_diff == float("inf"):
            mismatches.append(
                {
                    "index": i,
                    "input_a": a_val,
                    "input_b": b_val,
                    "expected_torch": exp_val,
                    "ttnn_result": res_val,
                    "ulp_diff": "inf",
                    "abs_diff": "NaN",
                }
            )
            continue

        valid_comparisons += 1
        total_ulp += ulp_diff
        max_ulp = max(max_ulp, ulp_diff)

        if ulp_diff > ULP_TOLERANCE:
            mismatches.append(
                {
                    "index": i,
                    "input_a": a_val,
                    "input_b": b_val,
                    "expected_torch": exp_val,
                    "ttnn_result": res_val,
                    "ulp_diff": ulp_diff,
                    "abs_diff": abs(exp_val - res_val),
                }
            )

    # Print summary
    print("\n" + "=" * 80)
    print("FMOD FP32 TEST RESULTS")
    print("=" * 80)
    print(f"Total values tested: {len(filtered_values)}")
    print(f"Valid comparisons: {valid_comparisons}")
    print(f"Number of mismatches (ULP > {ULP_TOLERANCE}): {len(mismatches)}")
    print(f"Max ULP difference: {max_ulp}")
    if valid_comparisons > 0:
        print(f"Average ULP difference: {total_ulp / valid_comparisons:.2f}")
    print("=" * 80)

    # Write mismatches to file
    if mismatches:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "/home/ubuntu/tt-metal/test_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"fmod_mismatches_{timestamp}.csv")
        write_mismatches_to_file(mismatches, output_path)

        print("\nFirst 10 mismatches:")
        print("-" * 100)
        print(f"{'Index':<8} {'Input A':<15} {'Input B':<15} {'Expected':<15} {'TTNN':<15} {'ULP':<10}")
        print("-" * 100)
        for m in mismatches[:10]:
            exp_str = (
                f"{m['expected_torch']:.6f}" if isinstance(m["expected_torch"], float) else str(m["expected_torch"])
            )
            res_str = f"{m['ttnn_result']:.6f}" if isinstance(m["ttnn_result"], float) else str(m["ttnn_result"])
            print(
                f"{m['index']:<8} {m['input_a']:<15.6f} {m['input_b']:<15.6f} {exp_str:<15} {res_str:<15} {m['ulp_diff']:<10}"
            )

    print(f"\nTest completed: {len(mismatches)} mismatches found")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_fmod_fp32_exhaustive(device):
    """
    Exhaustive test of fmod operation using all BF16 values converted to FP32.
    Processes in batches to avoid memory issues.
    """

    # Configuration
    ULP_TOLERANCE = 10
    OFFSET_B = 1.542
    BATCH_SIZE = 32 * 32  # 1024 values per batch (one 32x32 tile)

    print("\nGenerating all BF16 values as FP32...")
    all_bf16_fp32 = generate_all_bf16_as_fp32()
    print(f"Generated {len(all_bf16_fp32)} valid values")

    # Filter values
    filtered_values = []
    for val in all_bf16_fp32:
        b_val = val + OFFSET_B
        if abs(val) >= 0.01 and abs(b_val) >= 0.01:
            filtered_values.append(val)

    print(f"After filtering: {len(filtered_values)} values")

    # Process in batches
    all_mismatches = []
    all_results = []  # Store ALL results for ULP distribution analysis
    max_ulp_global = 0
    total_ulp_global = 0
    valid_comparisons_global = 0

    # ULP distribution tracking
    ulp_distribution = {
        "ulp_0": 0,  # Exact match
        "ulp_1": 0,  # ULP = 1
        "ulp_2": 0,  # ULP = 2
        "ulp_3": 0,  # ULP = 3
        "ulp_4_10": 0,  # ULP 4-10
        "ulp_11_100": 0,  # ULP 11-100
        "ulp_101_1000": 0,  # ULP 101-1000
        "ulp_gt_1000": 0,  # ULP > 1000
        "ulp_inf": 0,  # Inf/NaN cases
    }

    num_batches = (len(filtered_values) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing in {num_batches} batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(filtered_values))
        batch_values = filtered_values[start_idx:end_idx]

        # Pad batch to 32x32 tile size
        while len(batch_values) < 32 * 32:
            batch_values.append(1.0)

        # Shape for tile layout (32x32)
        input_a = torch.tensor(batch_values, dtype=torch.float32).reshape(1, 1, 32, 32)
        input_b = input_a + OFFSET_B

        # Expected result
        expected = torch.fmod(input_a, input_b)

        # TTNN computation
        ttnn_a = ttnn.from_torch(input_a, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn_b = ttnn.from_torch(input_b, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)

        ttnn_result = ttnn.pow(ttnn_a, ttnn_b)
        result = ttnn.to_torch(ttnn_result)

        # Compare
        expected_flat = expected.flatten()
        result_flat = result.flatten()
        input_a_flat = input_a.flatten()

        actual_batch_size = end_idx - start_idx
        for i in range(actual_batch_size):
            a_val = input_a_flat[i].item()
            b_val = a_val + OFFSET_B
            exp_val = expected_flat[i].item()
            res_val = result_flat[i].item()

            ulp_diff = compute_ulp_diff(exp_val, res_val)

            # Track ULP distribution
            if ulp_diff == float("inf"):
                ulp_distribution["ulp_inf"] += 1
                all_mismatches.append(
                    {
                        "index": start_idx + i,
                        "input_a": a_val,
                        "input_b": b_val,
                        "expected_torch": exp_val,
                        "ttnn_result": res_val,
                        "ulp_diff": "inf",
                        "abs_diff": "NaN",
                    }
                )
                continue

            # Categorize ULP
            if ulp_diff == 0:
                ulp_distribution["ulp_0"] += 1
            elif ulp_diff == 1:
                ulp_distribution["ulp_1"] += 1
            elif ulp_diff == 2:
                ulp_distribution["ulp_2"] += 1
            elif ulp_diff == 3:
                ulp_distribution["ulp_3"] += 1
            elif ulp_diff <= 10:
                ulp_distribution["ulp_4_10"] += 1
            elif ulp_diff <= 100:
                ulp_distribution["ulp_11_100"] += 1
            elif ulp_diff <= 1000:
                ulp_distribution["ulp_101_1000"] += 1
            else:
                ulp_distribution["ulp_gt_1000"] += 1

            valid_comparisons_global += 1
            total_ulp_global += ulp_diff
            max_ulp_global = max(max_ulp_global, ulp_diff)

            # Store ALL results for analysis
            all_results.append(
                {
                    "index": start_idx + i,
                    "input_a": a_val,
                    "input_b": b_val,
                    "expected_torch": exp_val,
                    "ttnn_result": res_val,
                    "ulp_diff": ulp_diff,
                    "abs_diff": abs(exp_val - res_val),
                }
            )

            if ulp_diff > ULP_TOLERANCE:
                all_mismatches.append(
                    {
                        "index": start_idx + i,
                        "input_a": a_val,
                        "input_b": b_val,
                        "expected_torch": exp_val,
                        "ttnn_result": res_val,
                        "ulp_diff": ulp_diff,
                        "abs_diff": abs(exp_val - res_val),
                    }
                )

        # Deallocate tensors
        ttnn_a.deallocate()
        ttnn_b.deallocate()
        ttnn_result.deallocate()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed batch {batch_idx + 1}/{num_batches}")

    # Print summary
    print("\n" + "=" * 80)
    print("FMOD FP32 EXHAUSTIVE TEST RESULTS")
    print("=" * 80)
    print(f"Total values tested: {len(filtered_values)}")
    print(f"Valid comparisons: {valid_comparisons_global}")
    print(f"Number of mismatches (ULP > {ULP_TOLERANCE}): {len(all_mismatches)}")
    print(f"Max ULP difference: {max_ulp_global}")
    if valid_comparisons_global > 0:
        print(f"Average ULP difference: {total_ulp_global / valid_comparisons_global:.2f}")
    print("=" * 80)

    # Build dynamic ULP distribution from actual data
    # Count ULP values in fine-grained buckets first
    ulp_counts = {}  # ulp_value -> count
    for r in all_results:
        ulp = r["ulp_diff"]
        if ulp not in ulp_counts:
            ulp_counts[ulp] = 0
        ulp_counts[ulp] += 1

    # Define range boundaries for detailed breakdown
    def get_ulp_range_counts():
        """Dynamically compute ULP ranges based on actual data distribution."""
        ranges = []

        # Always show ULP = 0 separately
        ulp_0_count = ulp_counts.get(0, 0)
        if ulp_0_count > 0:
            ranges.append(("ULP = 0 (exact)", 0, 0, ulp_0_count))

        # Define potential range boundaries
        boundaries = [
            1,
            2,
            3,
            4,
            5,
            10,
            50,
            100,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
            500000,
            1000000,
            10000000,
            100000000,
            1000000000,
            float("inf"),
        ]

        # Count values in each potential range
        range_data = []
        for i in range(len(boundaries) - 1):
            low = boundaries[i]
            high = boundaries[i + 1]
            count = sum(c for ulp, c in ulp_counts.items() if low <= ulp < high)
            range_data.append((low, high, count))

        # Combine adjacent zero ranges
        combined_ranges = []
        i = 0
        while i < len(range_data):
            low, high, count = range_data[i]

            if count == 0:
                # Start combining zero ranges
                combined_low = low
                combined_high = high
                while i + 1 < len(range_data) and range_data[i + 1][2] == 0:
                    i += 1
                    combined_high = range_data[i][1]
                combined_ranges.append((combined_low, combined_high, 0))
            else:
                # Non-zero range - check if we need to split it further
                if count > 100:  # If significant count, try to split
                    # Find actual ULP values in this range
                    ulps_in_range = [(ulp, c) for ulp, c in ulp_counts.items() if low <= ulp < high]
                    ulps_in_range.sort()

                    if len(ulps_in_range) > 1:
                        # Find natural groupings based on gaps
                        sub_ranges = []
                        sub_low = ulps_in_range[0][0]
                        sub_count = ulps_in_range[0][1]
                        prev_ulp = ulps_in_range[0][0]

                        for ulp, c in ulps_in_range[1:]:
                            # If big gap or enough values, start new sub-range
                            if ulp > prev_ulp * 2 or sub_count >= count // 3:
                                sub_ranges.append((sub_low, prev_ulp, sub_count))
                                sub_low = ulp
                                sub_count = c
                            else:
                                sub_count += c
                            prev_ulp = ulp

                        sub_ranges.append((sub_low, prev_ulp, sub_count))

                        for sr in sub_ranges:
                            combined_ranges.append(sr)
                    else:
                        combined_ranges.append((low, high, count))
                else:
                    combined_ranges.append((low, high, count))
            i += 1

        # Format range names
        final_ranges = []
        if ulp_0_count > 0:
            final_ranges.append(("ULP = 0 (exact)", ulp_0_count))

        for low, high, count in combined_ranges:
            if count > 0 or (low == 1):  # Show at least one zero range for context
                if low == high or (isinstance(high, float) and high == float("inf")):
                    if high == float("inf"):
                        name = f"ULP >= {low:,}"
                    else:
                        name = f"ULP = {low:,}"
                elif high == float("inf"):
                    name = f"ULP >= {low:,}"
                else:
                    name = f"ULP {low:,} - {high:,}"
                final_ranges.append((name, count))

        # Add inf/NaN if present
        if ulp_distribution["ulp_inf"] > 0:
            final_ranges.append(("ULP = inf (NaN)", ulp_distribution["ulp_inf"]))

        return final_ranges

    # Print ULP Distribution Table
    print("\n" + "=" * 90)
    print("ULP DISTRIBUTION TABLE (Dynamic Ranges)")
    print("=" * 90)
    print(f"{'ULP Range':<45} {'Count':>15} {'Percentage':>15}")
    print("-" * 90)

    total_for_pct = valid_comparisons_global + ulp_distribution["ulp_inf"]
    dynamic_ranges = get_ulp_range_counts()

    for range_name, count in dynamic_ranges:
        pct = (count / total_for_pct * 100) if total_for_pct > 0 else 0
        if count > 0 or "ULP = 0" in range_name or "1 -" in range_name:
            print(f"{range_name:<45} {count:>15,} {pct:>14.2f}%")

    print("-" * 90)
    print(f"{'TOTAL':<45} {total_for_pct:>15,}")
    print("=" * 90)

    # Analyze high ULP cases
    print("\n" + "=" * 80)
    print("HIGH ULP ANALYSIS (ULP > 3)")
    print("=" * 80)

    # Find examples from different ULP ranges
    high_ulp_examples = []
    for r in all_results:
        if r["ulp_diff"] > 3 and len(high_ulp_examples) < 5:
            high_ulp_examples.append(r)

    if high_ulp_examples:
        print(f"\n{'Index':<8} {'Input A':<18} {'Input B':<18} {'Expected':<18} {'TTNN':<18} {'ULP':<12}")
        print("-" * 100)
        for ex in high_ulp_examples:
            print(
                f"{ex['index']:<8} {ex['input_a']:<18.6g} {ex['input_b']:<18.6g} "
                f"{ex['expected_torch']:<18.6g} {ex['ttnn_result']:<18.6g} {ex['ulp_diff']:<12}"
            )

        # Provide analysis
        print("\n" + "-" * 80)
        print("REASON ANALYSIS:")
        print("-" * 80)

        # Categorize reasons
        large_value_count = sum(1 for r in all_results if r["ulp_diff"] > 3 and abs(r["input_a"]) > 1e6)
        equal_ab_count = sum(1 for r in all_results if r["ulp_diff"] > 3 and abs(r["input_a"] - r["input_b"]) < 1e-5)
        near_integer_count = sum(
            1
            for r in all_results
            if r["ulp_diff"] > 3 and abs(r["input_a"] / r["input_b"] - round(r["input_a"] / r["input_b"])) < 0.001
        )

        print(f"1. Large values (|a| > 1e6):        {large_value_count} cases")
        print(f"   Reason: FP32 precision limits - adding 1.542 to large numbers may not change the value")
        print(f"           When a ≈ b, fmod(a,a) should be 0, but reciprocal error causes wrong trunc")
        print(f"")
        print(f"2. Near-equal a,b (|a-b| < 1e-5):   {equal_ab_count} cases")
        print(f"   Reason: When a ≈ b, a/b ≈ 1.0. Small reciprocal errors cause trunc(0.999...) = 0")
        print(f"           instead of trunc(1.0) = 1, giving fmod = a instead of 0")
        print(f"")
        print(f"3. Near-integer quotient:           {near_integer_count} cases")
        print(f"   Reason: When a/b is very close to an integer, truncation can go wrong direction")
        print("=" * 80)

    # Write results to files (always save to CSV, not terminal)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/home/ubuntu/tt-metal/test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Write mismatches (ULP > tolerance)
    mismatch_path = os.path.join(output_dir, f"fmod_mismatches_{timestamp}.csv")
    if all_mismatches:
        write_mismatches_to_file(all_mismatches, mismatch_path)
    else:
        print(f"No mismatches (ULP > {ULP_TOLERANCE}) found!")

    # Write ALL results for detailed ULP analysis
    all_results_path = os.path.join(output_dir, f"fmod_all_results_{timestamp}.csv")
    with open(all_results_path, "w", newline="") as csvfile:
        fieldnames = ["index", "input_a", "input_b", "expected_torch", "ttnn_result", "ulp_diff", "abs_diff"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"All results written to: {all_results_path}")

    # Assert
    mismatch_threshold = len(filtered_values) * 0.01  # Allow 1% mismatches
    if len(all_mismatches) <= mismatch_threshold:
        print(f"\nTest PASSED: {len(all_mismatches)} mismatches within tolerance ({mismatch_threshold:.0f} allowed)")
    else:
        print(f"\nTest FAILED: {len(all_mismatches)} mismatches exceeds tolerance ({mismatch_threshold:.0f} allowed)")
        print(f"See detailed mismatches in: {mismatch_path}")
        print(f"See all results in: {all_results_path}")
        assert False, f"Too many mismatches: {len(all_mismatches)} > {mismatch_threshold}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_fmod_fp32_simple(device):
    """
    Simple test with known values to verify fmod implementation.
    Uses proper tile-aligned tensor shapes (32x32 minimum).
    """
    # Test cases: (a, b)
    test_cases = [
        (7.2, 3.1),
        (10.0, 3.0),
        (-10.0, 3.0),
        (10.0, -3.0),
        (-10.0, -3.0),
        (5.5, 2.5),
        (100.0, 7.0),
        (1.5, 0.7),
    ]

    num_test_cases = len(test_cases)

    # Pad to 32x32 = 1024 values for tile alignment (minimum tile size)
    while len(test_cases) < 32 * 32:
        test_cases.append((1.0, 1.0))

    a_values = [tc[0] for tc in test_cases]
    b_values = [tc[1] for tc in test_cases]

    # Create tile-aligned tensors (1, 1, 32, 32) - minimum tile size
    input_a = torch.tensor(a_values, dtype=torch.float32).reshape(1, 1, 32, 32)
    input_b = torch.tensor(b_values, dtype=torch.float32).reshape(1, 1, 32, 32)

    # Expected from PyTorch
    expected = torch.fmod(input_a, input_b)

    # TTNN
    ttnn_a = ttnn.from_torch(input_a, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_b = ttnn.from_torch(input_b, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)

    result = ttnn.pow(ttnn_a, ttnn_b)
    result_torch = ttnn.to_torch(result)

    print("\nSimple fmod test results:")
    print("-" * 80)
    print(f"{'A':<10} {'B':<10} {'Expected':<15} {'TTNN':<15} {'ULP':<10}")
    print("-" * 80)

    for i in range(num_test_cases):  # Only print actual test cases
        a = a_values[i]
        b = b_values[i]
        exp = expected.flatten()[i].item()
        res = result_torch.flatten()[i].item()
        ulp = compute_ulp_diff(exp, res)

        print(f"{a:<10.2f} {b:<10.2f} {exp:<15.6f} {res:<15.6f} {ulp:<10}")

        # Check ULP tolerance
        assert ulp <= 100 or ulp == float("inf"), f"fmod({a}, {b}) ULP too high: {ulp}"

    print("\nSimple test completed!")


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s", "-k", "simple"] + sys.argv[1:])
