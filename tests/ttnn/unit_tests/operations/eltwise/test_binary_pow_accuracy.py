# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import os
import csv
import pytest
import ttnn
import numpy as np
import matplotlib.pyplot as plt
from tests.ttnn.utils_for_testing import assert_with_ulp


def _run_exhaustive_pairwise_pow_helper(device, x_values, y_values, test_name, variant_name, max_skipped_samples=10):
    """Helper function to run exhaustive pairwise pow test between two value sets.

    Masks out invalid pow results:
    - NaN: x < 0 with non-integer y, or other undefined cases
    - ±Inf: overflow or 0^negative
    These are skipped from comparison since they represent mathematically undefined or
    implementation-sensitive cases.

    Returns:
        list: Up to max_skipped_samples skipped pairs as dicts with keys:
              [x, y, torch_result, ttnn_result, skip_reason, category]
    """
    ttnn_dtype = ttnn.bfloat16

    Nx = x_values.numel()
    Ny = y_values.numel()
    print(f"\n{'='*60}")
    print(f"{test_name}: Testing {Nx} × {Ny} = {Nx*Ny:,} pairs")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_pairs = 0
    total_skipped_pairs = 0

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Collect up to max_skipped_samples skipped pairs
    skipped_samples = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch = x_values[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = y_values.unsqueeze(0)  # shape: [1, Ny]

        # Broadcast pow
        z_torch = torch.pow(x_batch, y_full)  # shape: [B, Ny]

        # Send to backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.pow(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # ULP check (use bf16 space)
        z_bf16 = z_torch.to(torch.bfloat16).contiguous()
        tt_bf16 = tt_out.to(torch.bfloat16).contiguous()

        # Create validity mask: skip if torch result is NaN or ±Inf
        # These represent mathematically undefined inputs (x<0 with non-integer y)
        # or overflow/underflow cases (0^negative, large^large)
        valid_mask = torch.isfinite(z_bf16)
        skipped_mask = ~valid_mask
        batch_skipped = skipped_mask.sum().item()
        total_skipped_pairs += batch_skipped

        # Collect skipped samples (up to max_skipped_samples)
        if batch_skipped > 0 and len(skipped_samples) < max_skipped_samples:
            skipped_indices = skipped_mask.nonzero(as_tuple=False)
            for idx_pair in skipped_indices:
                if len(skipped_samples) >= max_skipped_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                torch_val = z_bf16[i, j].item()
                ttnn_val = tt_bf16[i, j].item()
                # Determine skip reason
                if torch.isnan(z_bf16[i, j]):
                    reason = "NaN"
                elif torch.isinf(z_bf16[i, j]):
                    reason = "+Inf" if z_bf16[i, j] > 0 else "-Inf"
                else:
                    reason = "Unknown"
                skipped_samples.append(
                    {
                        "x": xv,
                        "y": yv,
                        "torch_result": torch_val,
                        "ttnn_result": ttnn_val,
                        "skip_reason": reason,
                        "category": variant_name,
                    }
                )

        # Flush subnormal bfloat16 values to zero (only for valid comparisons)
        min_normal_threshold = torch.finfo(torch.bfloat16).tiny
        subnormal_mask = z_bf16.abs() <= min_normal_threshold
        z_bf16[subnormal_mask] = 0.0
        subnormal_mask_tt = tt_bf16.abs() <= min_normal_threshold
        tt_bf16[subnormal_mask_tt] = 0.0

        # For ttnn output, also flush non-finite to 0 for comparison purposes
        # (if torch was finite but ttnn overflowed, we want to detect that)
        tt_nonfinite_mask = ~torch.isfinite(tt_bf16)
        tt_bf16[tt_nonfinite_mask] = 0.0

        z_bits = z_bf16.view(torch.uint16).to(torch.int32)
        tt_bits = tt_bf16.view(torch.uint16).to(torch.int32)

        sign_z = (z_bits >> 15) & 1
        sign_tt = (tt_bits >> 15) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x8000, 0x8000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x8000, 0x8000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        # Apply validity mask - set ULP to 0 for invalid pairs (they will be skipped)
        ulp_dist_valid = torch.where(valid_mask, ulp_dist, torch.zeros_like(ulp_dist))

        max_ulp_batch = ulp_dist_valid.max().item()
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        # Count ULP distribution (only for valid pairs)
        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()

        # Count mismatches (only among valid pairs)
        mismatch_mask = (ulp_dist > 1) & valid_mask
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count
        total_valid_pairs += valid_mask.sum().item()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, skipped={batch_skipped}"
            )

    total_pairs = Nx * Ny
    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    skipped_pct = (total_skipped_pairs / total_pairs) * 100 if total_pairs > 0 else 0.0

    print(f"\n--- Summary ---")
    print(f"Total pairs: {total_pairs:,}")
    print(f"Valid pairs (torch finite): {total_valid_pairs:,} ({100-skipped_pct:.4f}%)")
    print(f"Skipped pairs (torch NaN/Inf): {total_skipped_pairs:,} ({skipped_pct:.4f}%)")
    print(f"Skipped samples collected: {len(skipped_samples)}")
    print(f"Global max ULP: {max_ulp_global}")
    print(f"Total mismatches (ULP > 1): {total_mismatches:,} ({mismatch_pct:.4f}% of valid)")

    print(f"\nULP Distribution (valid pairs only):")
    print(
        f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 0: 0"
    )
    print(
        f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 1: 0"
    )
    print(
        f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 2: 0"
    )
    print(
        f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 3-10: 0"
    )
    print(
        f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 11-100: 0"
    )
    print(
        f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP > 100: 0"
    )

    return skipped_samples


def _run_exhaustive_pairwise_pow_helper_float32(
    device, x_values, y_values, test_name, variant_name, max_skipped_samples=10
):
    """Helper function to run exhaustive pairwise pow test in float32 precision.

    Takes bfloat16 bit patterns as input, converts to float32, and performs pow in float32.
    Computes ULP distance in float32 space.

    Masks out invalid pow results:
    - NaN: x < 0 with non-integer y, or other undefined cases
    - ±Inf: overflow or 0^negative
    These are skipped from comparison since they represent mathematically undefined or
    implementation-sensitive cases.

    Returns:
        list: Up to max_skipped_samples skipped pairs as dicts with keys:
              [x, y, torch_result, ttnn_result, skip_reason, category]
    """
    ttnn_dtype = ttnn.float32

    Nx = x_values.numel()
    Ny = y_values.numel()
    print(f"\n{'='*60}")
    print(f"{test_name} (float32): Testing {Nx} × {Ny} = {Nx*Ny:,} pairs")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_pairs = 0
    total_skipped_pairs = 0

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Collect up to max_skipped_samples skipped pairs
    skipped_samples = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # Convert bfloat16 values to float32 for computation
        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch_f32 = x_values[start:end].to(torch.float32).unsqueeze(1)  # shape: [B, 1]
        y_full_f32 = y_values.to(torch.float32).unsqueeze(0)  # shape: [1, Ny]

        # Broadcast pow in float32
        z_torch = torch.pow(x_batch_f32, y_full_f32)  # shape: [B, Ny]

        # Send to backend using float32
        x_tt = ttnn.from_torch(x_batch_f32, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full_f32, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.pow(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # ULP check (use float32 space)
        z_f32 = z_torch.to(torch.float32).contiguous()
        tt_f32 = tt_out.to(torch.float32).contiguous()

        # Create validity mask: skip if torch result is NaN or ±Inf
        # These represent mathematically undefined inputs (x<0 with non-integer y)
        # or overflow/underflow cases (0^negative, large^large)
        valid_mask = torch.isfinite(z_f32)
        skipped_mask = ~valid_mask
        batch_skipped = skipped_mask.sum().item()
        total_skipped_pairs += batch_skipped

        # Collect skipped samples (up to max_skipped_samples)
        if batch_skipped > 0 and len(skipped_samples) < max_skipped_samples:
            skipped_indices = skipped_mask.nonzero(as_tuple=False)
            for idx_pair in skipped_indices:
                if len(skipped_samples) >= max_skipped_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                xv = x_batch_f32[i, 0].item()
                yv = y_full_f32[0, j].item()
                torch_val = z_f32[i, j].item()
                ttnn_val = tt_f32[i, j].item()
                # Determine skip reason
                if torch.isnan(z_f32[i, j]):
                    reason = "NaN"
                elif torch.isinf(z_f32[i, j]):
                    reason = "+Inf" if z_f32[i, j] > 0 else "-Inf"
                else:
                    reason = "Unknown"
                skipped_samples.append(
                    {
                        "x": xv,
                        "y": yv,
                        "torch_result": torch_val,
                        "ttnn_result": ttnn_val,
                        "skip_reason": reason,
                        "category": variant_name,
                    }
                )

        # Flush subnormal float32 values to zero (only for valid comparisons)
        min_normal_threshold = torch.finfo(torch.float32).tiny
        subnormal_mask = z_f32.abs() <= min_normal_threshold
        z_f32[subnormal_mask] = 0.0
        subnormal_mask_tt = tt_f32.abs() <= min_normal_threshold
        tt_f32[subnormal_mask_tt] = 0.0

        # For ttnn output, also flush non-finite to 0 for comparison purposes
        # (if torch was finite but ttnn overflowed, we want to detect that)
        tt_nonfinite_mask = ~torch.isfinite(tt_f32)
        tt_f32[tt_nonfinite_mask] = 0.0

        z_bits = z_f32.view(torch.int32)
        tt_bits = tt_f32.view(torch.int32)

        sign_z = (z_bits >> 31) & 1
        sign_tt = (tt_bits >> 31) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x80000000, 0x80000000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x80000000, 0x80000000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        # Apply validity mask - set ULP to 0 for invalid pairs (they will be skipped)
        ulp_dist_valid = torch.where(valid_mask, ulp_dist, torch.zeros_like(ulp_dist))

        max_ulp_batch = ulp_dist_valid.max().item()
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        # Count ULP distribution (only for valid pairs)
        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()

        # Count mismatches (only among valid pairs)
        mismatch_mask = (ulp_dist > 1) & valid_mask
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count
        total_valid_pairs += valid_mask.sum().item()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, skipped={batch_skipped}"
            )

    total_pairs = Nx * Ny
    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    skipped_pct = (total_skipped_pairs / total_pairs) * 100 if total_pairs > 0 else 0.0

    print(f"\n--- Summary ---")
    print(f"Total pairs: {total_pairs:,}")
    print(f"Valid pairs (torch finite): {total_valid_pairs:,} ({100-skipped_pct:.4f}%)")
    print(f"Skipped pairs (torch NaN/Inf): {total_skipped_pairs:,} ({skipped_pct:.4f}%)")
    print(f"Skipped samples collected: {len(skipped_samples)}")
    print(f"Global max ULP: {max_ulp_global}")
    print(f"Total mismatches (ULP > 1): {total_mismatches:,} ({mismatch_pct:.4f}% of valid)")

    print(f"\nULP Distribution (valid pairs only):")
    print(
        f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 0: 0"
    )
    print(
        f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 1: 0"
    )
    print(
        f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 2: 0"
    )
    print(
        f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 3-10: 0"
    )
    print(
        f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 11-100: 0"
    )
    print(
        f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP > 100: 0"
    )

    return skipped_samples


# FP32 exhaustive tests
def test_pow_tt_FP32_exhaustive_pos_pos(device):
    """Test positive base ^ positive exponent normal fp32 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, positive normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)

    pos_values = vals[mask]  # ~32512 positive normal values
    return _run_exhaustive_pairwise_pow_helper_float32(device, pos_values, pos_values, "Positive ^ Positive", "pos_pos")


def test_pow_tt_FP32_exhaustive_pos_neg(device):
    """Test positive base ^ negative exponent normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    pos_values = vals[pos_mask]  # ~32512 positive normal values
    neg_values = vals[neg_mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_pow_helper_float32(device, pos_values, neg_values, "Positive ^ Negative", "pos_neg")


def test_pow_tt_FP32_exhaustive_neg_pos(device):
    """Test negative base ^ positive exponent normal bf16 values.

    Note: This will have many NaN results since pow(negative, non-integer) is undefined.
    The masking in the helper function will skip these invalid cases.
    """
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    pos_values = vals[pos_mask]  # ~32512 positive normal values
    neg_values = vals[neg_mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_pow_helper_float32(device, neg_values, pos_values, "Negative ^ Positive", "neg_pos")


def test_pow_tt_FP32_exhaustive_neg_neg(device):
    """Test negative base ^ negative exponent normal bf16 values.

    Note: This will have many NaN results since pow(negative, non-integer) is undefined.
    The masking in the helper function will skip these invalid cases.
    """
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, negative normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    neg_values = vals[mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_pow_helper_float32(device, neg_values, neg_values, "Negative ^ Negative", "neg_neg")


def test_pow_tt_FP32_exhaustive_test(device):
    """Test pow with fp32 values - all sign combinations."""
    import io
    import sys
    from datetime import datetime

    # Tee class to write to both terminal and buffer
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    # Capture stdout while still printing to terminal
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, captured_output)

    # Collect skipped samples from all 4 categories
    all_skipped_samples = []

    try:
        all_skipped_samples.extend(test_pow_tt_FP32_exhaustive_pos_pos(device))
        all_skipped_samples.extend(test_pow_tt_FP32_exhaustive_pos_neg(device))
        all_skipped_samples.extend(test_pow_tt_FP32_exhaustive_neg_pos(device))
        all_skipped_samples.extend(test_pow_tt_FP32_exhaustive_neg_neg(device))
    finally:
        sys.stdout = original_stdout

    # Get captured content
    output_content = captured_output.getvalue()

    # Write results to file
    out_dir = "/home/ubuntu/Files"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"pow_fp32_exhaustive_results_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(output_content)

    # Write merged skipped samples CSV (40 rows max: 10 per category)
    skipped_csv_path = os.path.join(out_dir, f"pow_fp32_skipped_samples_{timestamp}.csv")
    with open(skipped_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "x", "y", "torch_result", "ttnn_result", "skip_reason"])
        for sample in all_skipped_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["x"],
                    sample["y"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["skip_reason"],
                ]
            )

    print(f"\n{'='*60}")
    print(f"All results saved to: {out_path}")
    print(f"Skipped samples CSV ({len(all_skipped_samples)} rows): {skipped_csv_path}")
    print(f"{'='*60}")


# BF16 Exhaustive tests
def test_pow_tt_bf16_exhaustive_pos_pos(device):
    """Test positive base ^ positive exponent normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, positive normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)

    pos_values = vals[mask]  # ~32512 positive normal values
    return _run_exhaustive_pairwise_pow_helper(device, pos_values, pos_values, "Positive ^ Positive", "pos_pos")


def test_pow_tt_bf16_exhaustive_pos_neg(device):
    """Test positive base ^ negative exponent normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    pos_values = vals[pos_mask]  # ~32512 positive normal values
    neg_values = vals[neg_mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_pow_helper(device, pos_values, neg_values, "Positive ^ Negative", "pos_neg")


def test_pow_tt_bf16_exhaustive_neg_pos(device):
    """Test negative base ^ positive exponent normal bf16 values.

    Note: This will have many NaN results since pow(negative, non-integer) is undefined.
    The masking in the helper function will skip these invalid cases.
    """
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    pos_values = vals[pos_mask]  # ~32512 positive normal values
    neg_values = vals[neg_mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_pow_helper(device, neg_values, pos_values, "Negative ^ Positive", "neg_pos")


def test_pow_tt_bf16_exhaustive_neg_neg(device):
    """Test negative base ^ negative exponent normal bf16 values.

    Note: This will have many NaN results since pow(negative, non-integer) is undefined.
    The masking in the helper function will skip these invalid cases.
    """
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, negative normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    neg_values = vals[mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_pow_helper(device, neg_values, neg_values, "Negative ^ Negative", "neg_neg")


def test_pow_tt_BF16_exhaustive_test(device):
    """Test pow with BF16 values - all sign combinations."""
    import io
    import sys
    from datetime import datetime

    # Tee class to write to both terminal and buffer
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    # Capture stdout while still printing to terminal
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, captured_output)

    # Collect skipped samples from all 4 categories
    all_skipped_samples = []

    try:
        all_skipped_samples.extend(test_pow_tt_bf16_exhaustive_pos_pos(device))
        all_skipped_samples.extend(test_pow_tt_bf16_exhaustive_pos_neg(device))
        all_skipped_samples.extend(test_pow_tt_bf16_exhaustive_neg_pos(device))
        all_skipped_samples.extend(test_pow_tt_bf16_exhaustive_neg_neg(device))
    finally:
        sys.stdout = original_stdout

    # Get captured content
    output_content = captured_output.getvalue()

    # Write results to file
    out_dir = "/home/ubuntu/Files"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"pow_bf16_exhaustive_results_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(output_content)

    # Write merged skipped samples CSV (40 rows max: 10 per category)
    skipped_csv_path = os.path.join(out_dir, f"pow_bf16_skipped_samples_{timestamp}.csv")
    with open(skipped_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "x", "y", "torch_result", "ttnn_result", "skip_reason"])
        for sample in all_skipped_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["x"],
                    sample["y"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["skip_reason"],
                ]
            )

    print(f"\n{'='*60}")
    print(f"All results saved to: {out_path}")
    print(f"Skipped samples CSV ({len(all_skipped_samples)} rows): {skipped_csv_path}")
    print(f"{'='*60}")
