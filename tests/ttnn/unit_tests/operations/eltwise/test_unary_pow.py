# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import csv
import os
from tests.ttnn.utils_for_testing import assert_with_ulp
from models.common.utility_functions import comp_ulp_check


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

    # Run comp_ulp_check to get ULP and mismatch info
    max_ulp, mismatch_file_path, failing_range = comp_ulp_check(
        input=tt_input,
        golden=golden,
        calculated=result,
        ulp_threshold=1,
        allow_nonfinite=True,
        input_b=tt_input,
        output_dir="binary_pow_mismatches.txt",
    )

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


def test_binary_pow_sweep(device):
    # Generate clean bf16 tensor (tensor A) - contains ~65k values (2^16 bf16 bit patterns)
    tensor_a = generate_clean_bf16_tensor()
    num_values = tensor_a.numel()  # Should be 65536 (2^16)

    # Output directory for mismatch files
    output_dir = "binary_pow_ulp_mismatches"
    os.makedirs(output_dir, exist_ok=True)

    # CSV file path
    csv_file_path = "binary_pow_results.csv"

    print(f"Testing binary pow with {num_values} B values...")
    print(f"Each iteration tests {num_values} element-wise A^B operations")

    # Open CSV file for writing results
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input_B", "Max_ULP", "Failing_Values"])

        # Iterate through each value in tensor_a as the B value
        for i in range(num_values):
            # Get the i-th value from tensor_a to use as B
            b_val = tensor_a[i]
            b_scalar = b_val.item()

            # Create tensor B filled with this single value (same shape as A: 65k elements)
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
            golden = torch.pow(tensor_a, tensor_b)

            # Run ttnn binary pow
            tt_result = ttnn.pow(tt_a, tt_b)
            result = ttnn.to_torch(tt_result)

            # Run comp_ulp_check to get ULP and mismatch info
            max_ulp, mismatch_file_path, failing_range = comp_ulp_check(
                input=tensor_a,
                golden=golden,
                calculated=result,
                ulp_threshold=1,
                allow_nonfinite=True,
                input_b=b_val,
                output_dir=output_dir,
            )

            # Only write to CSV if ULP > 1
            if max_ulp > 1:
                # Format failing values column
                if mismatch_file_path is not None and failing_range is not None:
                    failing_values = f"File: {mismatch_file_path}, Range: {failing_range}"
                else:
                    failing_values = "No mismatches"

                # Write to CSV
                csv_writer.writerow([b_scalar, max_ulp, failing_values])

            # Print progress every 1000 iterations
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_values} B values...")

    print(f"\nResults saved to {csv_file_path}")
    print(f"ULP mismatch details saved to {output_dir}/")


def main():
    """Main function to run binary pow sweep test standalone."""
    import ttnn

    # Initialize device
    device = ttnn.open_device(device_id=0)

    try:
        test_binary_pow_sweep(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
