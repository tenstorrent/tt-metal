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


def _run_exhaustive_pairwise_mul_helper(
    device, x_values, y_values, test_name, variant_name, max_skipped_samples=10, max_mismatch_samples=1000
):
    """Helper function to run exhaustive pairwise fmod test between two value sets.

    Filters out pairs where |x/y| >= 2^7 (128) since these exceed BF16 mantissa precision.

    Returns:
        tuple: (skipped_samples, mismatch_samples)
            - skipped_samples: Up to max_skipped_samples skipped pairs
            - mismatch_samples: Up to max_mismatch_samples ULP > 1 pairs
    """
    ttnn_dtype = ttnn.bfloat16
    quotient_threshold = 2**7  # 128 for BF16 (7-bit mantissa)

    Nx = x_values.numel()
    Ny = y_values.numel()
    print()
    print(f"{test_name}")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_pairs = 0  # Track total valid pairs after filtering
    total_filtered_pairs = 0  # Track filtered out pairs

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Collect up to max_skipped_samples skipped pairs
    skipped_samples = []
    # Collect up to max_mismatch_samples ULP > 1 pairs
    mismatch_samples = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch = x_values[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = y_values.unsqueeze(0)  # shape: [1, Ny]

        # Filter valid range: |x/y| < 2^7 for BF16 precision
        # This ensures the quotient can be represented accurately
        total_batch_pairs = x_batch.shape[0] * y_full.shape[1]

        with torch.no_grad():
            # Compute |x/y| safely (avoid division by zero)
            y_safe = torch.where(y_full == 0, torch.ones_like(y_full), y_full)
            quotient_abs = (x_batch / y_safe).abs()
            # Create valid mask (quotient within range and y is not zero)
            valid_mask = (quotient_abs < quotient_threshold) & (y_full != 0)
            skipped_mask = ~valid_mask

        valid_count = valid_mask.sum().item()
        batch_skipped = skipped_mask.sum().item()

        # Track valid/filtered pairs
        total_valid_pairs += valid_count
        total_filtered_pairs += batch_skipped

        # Compute fmod on full tensor (faster than extracting valid pairs)
        z_torch = torch.fmod(x_batch, y_full)  # shape: [B, Ny]

        # Send to backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.pow(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # Collect skipped samples (up to max_skipped_samples)
        if batch_skipped > 0 and len(skipped_samples) < max_skipped_samples:
            skipped_indices = skipped_mask.nonzero(as_tuple=False)
            for idx_pair in skipped_indices:
                if len(skipped_samples) >= max_skipped_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                x_div_y = xv / yv if yv != 0 else float("inf")
                torch_val = z_torch[i, j].item()
                ttnn_val = tt_out[i, j].to(torch.bfloat16).item()
                # Determine skip reason
                if yv == 0:
                    reason = "y=0 (division by zero)"
                elif abs(x_div_y) >= quotient_threshold:
                    reason = f"|x/y|={abs(x_div_y):.2f} >= 2^7 (exceeds BF16 mantissa precision)"
                else:
                    reason = "Unknown"
                skipped_samples.append(
                    {
                        "x": xv,
                        "y": yv,
                        "x_div_y": x_div_y,
                        "torch_result": torch_val,
                        "ttnn_result": ttnn_val,
                        "skip_reason": reason,
                        "category": variant_name,
                    }
                )

        # ULP check (use bf16 space)
        z_bf16 = z_torch.to(torch.bfloat16).contiguous()
        tt_bf16 = tt_out.to(torch.bfloat16).contiguous()

        # Flush subnormal bfloat16 values to zero
        # For bfloat16, min normal is ~1.175494e-38 (same exponent range as float32)
        min_normal_threshold = torch.finfo(torch.bfloat16).tiny
        subnormal_mask = z_bf16.abs() <= min_normal_threshold
        z_bf16[subnormal_mask] = 0.0
        subnormal_mask_tt = tt_bf16.abs() <= min_normal_threshold
        tt_bf16[subnormal_mask_tt] = 0.0

        # Flush overflow values (greater than max normal or inf) to zero
        # For bfloat16, max normal is ~3.389531e38 (same exponent range as float32)
        max_normal_threshold = torch.finfo(torch.bfloat16).max
        overflow_mask = (z_bf16.abs() > max_normal_threshold) | ~torch.isfinite(z_bf16)
        z_bf16[overflow_mask] = 0.0
        overflow_mask_tt = (tt_bf16.abs() > max_normal_threshold) | ~torch.isfinite(tt_bf16)
        tt_bf16[overflow_mask_tt] = 0.0

        z_bits = z_bf16.view(torch.uint16).to(torch.int32)
        tt_bits = tt_bf16.view(torch.uint16).to(torch.int32)

        sign_z = (z_bits >> 15) & 1
        sign_tt = (tt_bits >> 15) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x8000, 0x8000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x8000, 0x8000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        # Only consider valid pairs for ULP statistics (masking approach - faster)
        max_ulp_batch = ulp_dist[valid_mask].max().item() if valid_count > 0 else 0
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        # Count ULP distribution (only for valid pairs)
        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()

        # # Log mismatches to CSV
        # mismatch_mask = ulp_dist > 1
        # mismatch_count = mismatch_mask.sum().item()
        # total_mismatches += mismatch_count
        #
        # if mismatch_count > 0:
        #     # Get mismatch indices in [B, Ny] grid
        #     mismatch_indices = mismatch_mask.nonzero(as_tuple=False)  # shape: [num_mismatches, 2]
        #
        #     for idx_pair in mismatch_indices:
        #         i, j = idx_pair[0].item(), idx_pair[1].item()
        #         x_idx = start + i
        #         y_idx = j
        #
        #         xv = x_batch[i, 0].item()
        #         yv = y_full[0, j].item()
        #         zv = z_bf16[i, j].item()
        #         tv = tt_bf16[i, j].item()
        #         ulp = ulp_dist[i, j].item()
        #
        #         xb = z_bf16.new_tensor(xv).view(torch.uint16).item()
        #         yb = z_bf16.new_tensor(yv).view(torch.uint16).item()
        #         zb = z_bits[i, j].item() & 0xFFFF
        #         tb = tt_bits[i, j].item() & 0xFFFF
        #
        #         writer.writerow([batch_idx, x_idx, y_idx, xv, yv, zv, tv, ulp, xb, yb, zb, tb])

        mismatch_mask = (ulp_dist > 1) & valid_mask
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        # Collect mismatch samples (ULP > 1)
        if mismatch_count > 0 and len(mismatch_samples) < max_mismatch_samples:
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
            for idx_pair in mismatch_indices:
                if len(mismatch_samples) >= max_mismatch_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                torch_val = z_bf16[i, j].item()
                ttnn_val = tt_bf16[i, j].item()
                ulp_val = ulp_dist[i, j].item()
                mismatch_samples.append(
                    {
                        "x": xv,
                        "y": yv,
                        "torch_result": torch_val,
                        "ttnn_result": ttnn_val,
                        "ulp": ulp_val,
                        "category": variant_name,
                    }
                )

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, tested={valid_count}/{total_batch_pairs}"
            )

    total_pairs = Nx * Ny
    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Valid Range Filter: |x/y| < 2^7 (128) for BF16")
    print(f"Total input pairs: {total_pairs:,}")
    print(f"Valid input pairs (|x/y| < 2^7): {total_valid_pairs:,} ({total_valid_pairs/total_pairs*100:.4f}%)")
    print(f"Input pairs removed (|x/y| >= 2^7): {total_filtered_pairs:,} ({total_filtered_pairs/total_pairs*100:.4f}%)")
    print(f"{'='*60}")
    print(
        f"Global max ULP: {max_ulp_global}, total mismatches: {total_mismatches}/{total_valid_pairs} ({mismatch_pct:.4f}%)"
    )
    print(f"\nULP Distribution (valid pairs only):")
    print(
        f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 0: 0 (N/A)"
    )
    print(
        f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 1: 0 (N/A)"
    )
    print(
        f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 2: 0 (N/A)"
    )
    print(
        f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 3-10: 0 (N/A)"
    )
    print(
        f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 11-100: 0 (N/A)"
    )
    print(
        f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP > 100: 0 (N/A)"
    )
    print(f"Skipped samples collected: {len(skipped_samples)}")
    print(f"Mismatch samples collected (ULP > 1): {len(mismatch_samples)}")

    return skipped_samples, mismatch_samples


def _run_exhaustive_pairwise_mul_helper_float32(
    device, x_values, y_values, test_name, variant_name, max_skipped_samples=10, max_mismatch_samples=1000
):
    """Helper function to run exhaustive pairwise fmod test in float32 precision.

    Takes bfloat16 bit patterns as input, converts to float32, and performs fmod in float32.
    Computes ULP distance in float32 space.

    Filters out pairs where |x/y| >= 2^23 since these exceed FP32 mantissa precision.

    Returns:
        tuple: (skipped_samples, mismatch_samples)
            - skipped_samples: Up to max_skipped_samples skipped pairs
            - mismatch_samples: Up to max_mismatch_samples ULP > 1 pairs
    """
    ttnn_dtype = ttnn.float32
    quotient_threshold = 2**23  # 8,388,608 for FP32 (23-bit mantissa)

    Nx = x_values.numel()
    Ny = y_values.numel()
    print(f"\n{'='*60}")
    print(f"{test_name} (float32): Testing {Nx} × {Ny} = {Nx*Ny:,} pairs")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_pairs = 0  # Track total valid pairs after filtering
    total_filtered_pairs = 0  # Track filtered out pairs

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Collect up to max_skipped_samples skipped pairs
    skipped_samples = []
    # Collect up to max_mismatch_samples ULP > 1 pairs
    mismatch_samples = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # Convert bfloat16 values to float32 for computation
        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch_f32 = x_values[start:end].to(torch.float32).unsqueeze(1)  # shape: [B, 1]
        y_full_f32 = y_values.to(torch.float32).unsqueeze(0)  # shape: [1, Ny]

        # Filter valid range: |x/y| < 2^23 for FP32 precision
        # This ensures the quotient can be represented accurately
        total_batch_pairs = x_batch_f32.shape[0] * y_full_f32.shape[1]

        with torch.no_grad():
            # Compute |x/y| safely (avoid division by zero)
            y_safe = torch.where(y_full_f32 == 0, torch.ones_like(y_full_f32), y_full_f32)
            quotient_abs = (x_batch_f32 / y_safe).abs()
            # Create valid mask (quotient within range and y is not zero)
            valid_mask = (quotient_abs < quotient_threshold) & (y_full_f32 != 0)
            skipped_mask = ~valid_mask

        valid_count = valid_mask.sum().item()
        batch_skipped = skipped_mask.sum().item()

        # Track valid/filtered pairs
        total_valid_pairs += valid_count
        total_filtered_pairs += batch_skipped

        # Compute fmod on full tensor (faster than extracting valid pairs)
        z_torch = torch.fmod(x_batch_f32, y_full_f32)  # shape: [B, Ny]

        # Send to backend using float32
        x_tt = ttnn.from_torch(x_batch_f32, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full_f32, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.pow(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # Collect skipped samples (up to max_skipped_samples)
        if batch_skipped > 0 and len(skipped_samples) < max_skipped_samples:
            skipped_indices = skipped_mask.nonzero(as_tuple=False)
            for idx_pair in skipped_indices:
                if len(skipped_samples) >= max_skipped_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                xv = x_batch_f32[i, 0].item()
                yv = y_full_f32[0, j].item()
                x_div_y = xv / yv if yv != 0 else float("inf")
                torch_val = z_torch[i, j].item()
                ttnn_val = tt_out[i, j].item()
                # Determine skip reason
                if yv == 0:
                    reason = "y=0 (division by zero)"
                elif abs(x_div_y) >= quotient_threshold:
                    reason = f"|x/y|={abs(x_div_y):.2e} >= 2^23 (exceeds FP32 mantissa precision)"
                else:
                    reason = "Unknown"
                skipped_samples.append(
                    {
                        "x": xv,
                        "y": yv,
                        "x_div_y": x_div_y,
                        "torch_result": torch_val,
                        "ttnn_result": ttnn_val,
                        "skip_reason": reason,
                        "category": variant_name,
                    }
                )

        # ULP check (use float32 space)
        z_f32 = z_torch.to(torch.float32).contiguous()
        tt_f32 = tt_out.to(torch.float32).contiguous()

        # Flush subnormal float32 values to zero
        # For float32, min normal is ~1.175494e-38
        min_normal_threshold = torch.finfo(torch.float32).tiny
        subnormal_mask = z_f32.abs() <= min_normal_threshold
        z_f32[subnormal_mask] = 0.0
        subnormal_mask_tt = tt_f32.abs() <= min_normal_threshold
        tt_f32[subnormal_mask_tt] = 0.0

        # Flush overflow values (greater than max normal or inf) to zero
        # For float32, max normal is ~3.389531e38
        max_normal_threshold = torch.finfo(torch.float32).max
        overflow_mask = (z_f32.abs() > max_normal_threshold) | ~torch.isfinite(z_f32)
        z_f32[overflow_mask] = 0.0
        overflow_mask_tt = (tt_f32.abs() > max_normal_threshold) | ~torch.isfinite(tt_f32)
        tt_f32[overflow_mask_tt] = 0.0

        z_bits = z_f32.view(torch.int32)
        tt_bits = tt_f32.view(torch.int32)

        sign_z = (z_bits >> 31) & 1
        sign_tt = (tt_bits >> 31) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x80000000, 0x80000000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x80000000, 0x80000000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        # Only consider valid pairs for ULP statistics (masking approach - faster)
        max_ulp_batch = ulp_dist[valid_mask].max().item() if valid_count > 0 else 0
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        # Count ULP distribution (only for valid pairs)
        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()

        # Log mismatches (only for valid pairs)
        mismatch_mask = (ulp_dist > 1) & valid_mask
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        # Collect mismatch samples (ULP > 1)
        if mismatch_count > 0 and len(mismatch_samples) < max_mismatch_samples:
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
            for idx_pair in mismatch_indices:
                if len(mismatch_samples) >= max_mismatch_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                xv = x_batch_f32[i, 0].item()
                yv = y_full_f32[0, j].item()
                torch_val = z_f32[i, j].item()
                ttnn_val = tt_f32[i, j].item()
                ulp_val = ulp_dist[i, j].item()
                mismatch_samples.append(
                    {
                        "x": xv,
                        "y": yv,
                        "torch_result": torch_val,
                        "ttnn_result": ttnn_val,
                        "ulp": ulp_val,
                        "category": variant_name,
                    }
                )

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, tested={valid_count}/{total_batch_pairs}"
            )

    total_pairs = Nx * Ny
    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Valid Range Filter: |x/y| < 2^23 (8,388,608) for FP32")
    print(f"Total input pairs: {total_pairs:,}")
    print(f"Valid input pairs (|x/y| < 2^23): {total_valid_pairs:,} ({total_valid_pairs/total_pairs*100:.4f}%)")
    print(
        f"Input pairs removed (|x/y| >= 2^23): {total_filtered_pairs:,} ({total_filtered_pairs/total_pairs*100:.4f}%)"
    )
    print(f"{'='*60}")
    print(
        f"Global max ULP: {max_ulp_global}, total mismatches: {total_mismatches}/{total_valid_pairs} ({mismatch_pct:.4f}%)"
    )
    print(f"\nULP Distribution (valid pairs only):")
    print(
        f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 0: 0 (N/A)"
    )
    print(
        f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 1: 0 (N/A)"
    )
    print(
        f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP = 2: 0 (N/A)"
    )
    print(
        f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 3-10: 0 (N/A)"
    )
    print(
        f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP 11-100: 0 (N/A)"
    )
    print(
        f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_pairs*100:.4f}%)"
        if total_valid_pairs > 0
        else "  ULP > 100: 0 (N/A)"
    )
    print(f"Skipped samples collected: {len(skipped_samples)}")
    print(f"Mismatch samples collected (ULP > 1): {len(mismatch_samples)}")

    return skipped_samples, mismatch_samples


# FP32 exhaustive tests
def test_mul_tt_FP32_exhaustive_pos_pos(device):
    """Test positive × positive normal fp32 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, positive normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)

    pos_values = vals[mask]  # ~32512 positive normal values
    return _run_exhaustive_pairwise_mul_helper_float32(device, pos_values, pos_values, "Positive × Positive", "pos_pos")


def test_mul_tt_FP32_exhaustive_pos_neg(device):
    """Test positive × negative normal bf16 values."""
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
    return _run_exhaustive_pairwise_mul_helper_float32(device, pos_values, neg_values, "Positive × Negative", "pos_neg")


def test_mul_tt_FP32_exhaustive_neg_pos(device):
    """Test negative × positive normal bf16 values."""
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
    return _run_exhaustive_pairwise_mul_helper_float32(device, neg_values, pos_values, "Negative × Positive", "neg_pos")


def test_mul_tt_FP32_exhaustive_neg_neg(device):
    """Test negative × negative normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, negative normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    neg_values = vals[mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_mul_helper_float32(device, neg_values, neg_values, "Negative × Negative", "neg_neg")


def test_fmod_tt_FP32_exhaustive_test(device):
    """Test fmod with fp32 values - all sign combinations."""
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

    # Collect skipped and mismatch samples from all 4 categories
    all_skipped_samples = []
    all_mismatch_samples = []

    try:
        skipped, mismatches = test_mul_tt_FP32_exhaustive_pos_pos(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)

        skipped, mismatches = test_mul_tt_FP32_exhaustive_pos_neg(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)

        skipped, mismatches = test_mul_tt_FP32_exhaustive_neg_pos(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)

        skipped, mismatches = test_mul_tt_FP32_exhaustive_neg_neg(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)
    finally:
        sys.stdout = original_stdout

    # Get captured content
    output_content = captured_output.getvalue()

    # Write results to file
    out_dir = "/home/ubuntu/Files"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"fmod_fp32_exhaustive_results_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(output_content)

    # Write merged skipped samples CSV (40 rows max: 10 per category)
    skipped_csv_path = os.path.join(out_dir, f"fmod_fp32_skipped_samples_{timestamp}.csv")
    with open(skipped_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "x", "y", "x_div_y", "torch_result", "ttnn_result", "skip_reason"])
        for sample in all_skipped_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["x"],
                    sample["y"],
                    sample["x_div_y"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["skip_reason"],
                ]
            )

    # Write merged mismatch samples CSV (ULP > 1)
    mismatch_csv_path = os.path.join(out_dir, f"fmod_fp32_ulp_gt1_mismatches_{timestamp}.csv")
    with open(mismatch_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "x", "y", "torch_result", "ttnn_result", "ulp"])
        for sample in all_mismatch_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["x"],
                    sample["y"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["ulp"],
                ]
            )

    print(f"\n{'='*60}")
    print(f"All results saved to: {out_path}")
    print(f"Skipped samples CSV ({len(all_skipped_samples)} rows): {skipped_csv_path}")
    print(f"ULP > 1 mismatches CSV ({len(all_mismatch_samples)} rows): {mismatch_csv_path}")
    print(f"{'='*60}")


# BF16 Exhaustive tests
def test_fmod_tt_bf16_exhaustive_pos_pos(device):
    """Test positive × positive normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, positive normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)

    pos_values = vals[mask]  # ~32512 positive normal values
    return _run_exhaustive_pairwise_mul_helper(device, pos_values, pos_values, "Positive × Positive", "pos_pos")


def test_fmod_tt_bf16_exhaustive_pos_neg(device):
    """Test positive × negative normal bf16 values."""
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
    return _run_exhaustive_pairwise_mul_helper(device, pos_values, neg_values, "Positive × Negative", "pos_neg")


def test_fmod_tt_bf16_exhaustive_neg_pos(device):
    """Test negative × positive normal bf16 values."""
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
    return _run_exhaustive_pairwise_mul_helper(device, neg_values, pos_values, "Negative × Positive", "neg_pos")


def test_fmod_tt_bf16_exhaustive_neg_neg(device):
    """Test negative × negative normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite, non-zero, negative normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    neg_values = vals[mask]  # ~32512 negative normal values
    return _run_exhaustive_pairwise_mul_helper(device, neg_values, neg_values, "Negative × Negative", "neg_neg")


def test_fmod_tt_BF16_exhaustive_test(device):
    """Test fmod with BF16 values - all sign combinations."""
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

    # Collect skipped and mismatch samples from all 4 categories
    all_skipped_samples = []
    all_mismatch_samples = []

    try:
        skipped, mismatches = test_fmod_tt_bf16_exhaustive_pos_pos(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)

        skipped, mismatches = test_fmod_tt_bf16_exhaustive_pos_neg(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)

        skipped, mismatches = test_fmod_tt_bf16_exhaustive_neg_pos(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)

        skipped, mismatches = test_fmod_tt_bf16_exhaustive_neg_neg(device)
        all_skipped_samples.extend(skipped)
        all_mismatch_samples.extend(mismatches)
    finally:
        sys.stdout = original_stdout

    # Get captured content
    output_content = captured_output.getvalue()

    # Write results to file
    out_dir = "/home/ubuntu/Files"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"fmod_bf16_exhaustive_results_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(output_content)

    # Write merged skipped samples CSV (40 rows max: 10 per category)
    skipped_csv_path = os.path.join(out_dir, f"fmod_bf16_skipped_samples_{timestamp}.csv")
    with open(skipped_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "x", "y", "x_div_y", "torch_result", "ttnn_result", "skip_reason"])
        for sample in all_skipped_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["x"],
                    sample["y"],
                    sample["x_div_y"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["skip_reason"],
                ]
            )

    # Write merged mismatch samples CSV (ULP > 1)
    mismatch_csv_path = os.path.join(out_dir, f"fmod_bf16_ulp_gt1_mismatches_{timestamp}.csv")
    with open(mismatch_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "x", "y", "torch_result", "ttnn_result", "ulp"])
        for sample in all_mismatch_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["x"],
                    sample["y"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["ulp"],
                ]
            )

    print(f"\n{'='*60}")
    print(f"All results saved to: {out_path}")
    print(f"Skipped samples CSV ({len(all_skipped_samples)} rows): {skipped_csv_path}")
    print(f"ULP > 1 mismatches CSV ({len(all_mismatch_samples)} rows): {mismatch_csv_path}")
    print(f"{'='*60}")


def test_binary_fp32(device, low=-1e-20, high=1e20):
    torch.manual_seed(0)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.float32
    x_torch = torch.rand([1, 12288], dtype=torch_dtype) * (high - low) + low
    y_torch = torch.rand([1, 1], dtype=torch_dtype) * (high - low) + low
    z_torch = torch.mul(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.mul(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt)

    # Calculate ULP distance
    z_f32 = z_torch.to(torch.float32).contiguous()
    tt_f32 = tt_out.to(torch.float32).contiguous()

    # Check for non-finite values first
    z_nonfinite = ~torch.isfinite(z_f32)
    tt_nonfinite = ~torch.isfinite(tt_f32)
    nonfinite_count = (z_nonfinite | tt_nonfinite).sum().item()

    print(f"\n=== Diagnostics ===")
    print(f"Total elements: {z_f32.numel()}")
    print(f"Non-finite in torch output: {z_nonfinite.sum().item()}")
    print(f"Non-finite in ttnn output: {tt_nonfinite.sum().item()}")
    print(f"Total non-finite: {nonfinite_count}")

    # Print some example values
    print(f"\nInput x_torch range: [{x_torch.min().item():.6e}, {x_torch.max().item():.6e}]")
    print(f"Input y_torch range: [{y_torch.min().item():.6e}, {y_torch.max().item():.6e}]")
    print(
        f"Output z_torch range: [{z_f32[torch.isfinite(z_f32)].min().item() if torch.isfinite(z_f32).any() else float('nan'):.6e}, {z_f32[torch.isfinite(z_f32)].max().item() if torch.isfinite(z_f32).any() else float('nan'):.6e}] (finite only)"
    )
    print(
        f"Output tt_out range: [{tt_f32[torch.isfinite(tt_f32)].min().item() if torch.isfinite(tt_f32).any() else float('nan'):.6e}, {tt_f32[torch.isfinite(tt_f32)].max().item() if torch.isfinite(tt_f32).any() else float('nan'):.6e}] (finite only)"
    )

    # Print first 10 values
    print(f"\nFirst 10 x values: {x_torch[0, :10].tolist()}")
    print(f"First 10 y values: {y_torch[:10].tolist()}")
    print(f"First 10 z_torch values: {z_f32[0, :10].tolist()}")
    print(f"First 10 tt_out values: {tt_f32[0, :10].tolist()}")

    z_bits = z_f32.view(torch.int32)
    tt_bits = tt_f32.view(torch.int32)

    sign_z = (z_bits >> 31) & 1
    sign_tt = (tt_bits >> 31) & 1
    z_ord = torch.where(sign_z == 0, z_bits + 0x80000000, 0x80000000 - z_bits)
    tt_ord = torch.where(sign_tt == 0, tt_bits + 0x80000000, 0x80000000 - tt_bits)
    ulp_dist = (z_ord - tt_ord).abs()

    # Find mismatches (ULP > 1) - only check finite values
    finite_mask = torch.isfinite(z_f32) & torch.isfinite(tt_f32)
    mismatch_mask = (ulp_dist > 1) & finite_mask
    mismatch_count = mismatch_mask.sum().item()

    print(f"\nFinite values: {finite_mask.sum().item()}")
    print(f"Mismatches (ULP > 1) in finite values: {mismatch_count}")

    if mismatch_count > 0:
        # Write mismatches to CSV
        out_dir = os.path.dirname(__file__)
        out_path = os.path.join(out_dir, "mul_binary_fp32_mismatchBH.csv")
        csv_file = open(out_path, mode="w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "index",
                "input_A",
                "input_B",
                "torch_output",
                "ttnn_output",
                "ulp_distance",
                "input_A_bits_u32",
                "input_B_bits_u32",
                "torch_output_bits_u32",
                "ttnn_output_bits_u32",
            ]
        )

        # Get mismatch indices
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)  # shape: [num_mismatches, 2]

        for idx_pair in mismatch_indices:
            i, j = idx_pair[0].item(), idx_pair[1].item()

            x_val = x_torch[i, j].item()
            y_val = y_torch[j].item()
            z_val = z_f32[i, j].item()
            tt_val = tt_f32[i, j].item()
            ulp = ulp_dist[i, j].item()

            # Get bit representations as uint32
            x_bits = z_f32.new_tensor(x_val).view(torch.int32).item() & 0xFFFFFFFF
            y_bits = z_f32.new_tensor(y_val).view(torch.int32).item() & 0xFFFFFFFF
            z_bits_val = z_bits[i, j].item() & 0xFFFFFFFF
            tt_bits_val = tt_bits[i, j].item() & 0xFFFFFFFF

            writer.writerow(
                [
                    f"({i},{j})",
                    x_val,
                    y_val,
                    z_val,
                    tt_val,
                    ulp,
                    hex(x_bits),
                    hex(y_bits),
                    hex(z_bits_val),
                    hex(tt_bits_val),
                ]
            )

        csv_file.close()
        print(f"Found {mismatch_count} mismatches (ULP > 1), written to {out_path}")
    else:
        print("No mismatches found (all ULP <= 1)")

    # torch.set_printoptions(linewidth=200, threshold = 10000 , precision=15, sci_mode = False, edgeitems=17)
    # print("z_tt", z_tt)
    # print("tt_out", tt_out)
    # print("z_torch", z_torch)

    # Allow non-finite values if both outputs agree on them
    assert_with_ulp(z_torch, tt_out, 1, allow_nonfinite=True)
