#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Python test for ttnn.add ULP accuracy
Run directly: python test_all_range_add.py

This test exhaustively covers ALL bfloat16 bit patterns (65,536 values)
and outputs detailed results to CSV for analysis.
"""

import torch
import ttnn
import csv
import json
from models.common.utility_functions import calculate_detailed_ulp_stats
from loguru import logger
import sys
import os
from tests.ttnn.utils_for_testing import assert_with_ulp
import pytest
import math
import matplotlib.pyplot as plt
import numpy as np


### -------------------- BF16 1e15 RANGE -------------------- ###


def test_add_tt_bf16_exhaustive_pairwise_1e15(device):
    torch.manual_seed(0)
    ttnn_dtype = ttnn.bfloat16

    # Generate all bf16 values in range [1e-15, 1e15] by magnitude
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    min_abs = torch.tensor(1e-15, dtype=torch.float32)
    max_abs = torch.tensor(1e15, dtype=torch.float32)
    vals_f32_abs = vals.to(torch.float32).abs()
    mask = torch.isfinite(vals) & (vals_f32_abs >= min_abs) & (vals_f32_abs <= max_abs)

    value_set = vals[mask]  # ~25510 values
    N = value_set.numel()
    print(f"Testing {N} values → {N}x{N} = {N*N:,} pairs")

    # Create all pairs via meshgrid (memory-intensive; consider batching)
    # For full test: split into chunks to avoid OOM

    batch_size = 512  # process 512×N pairs at a time
    num_batches = (N + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Open CSV for logging mismatches
    out_dir = os.path.dirname(__file__)
    out_path_mismatch = os.path.join(out_dir, "add_ulp_mismatch_exhaustive_1e15.csv")
    csv_file = open(out_path_mismatch, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "batch_idx",
            "x_idx",
            "y_idx",
            "x_value",
            "y_value",
            "torch_out",
            "tt_out",
            "ulp_distance",
            "x_bits_u16",
            "y_bits_u16",
            "torch_bits_u16",
            "tt_bits_u16",
        ]
    )

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        # x_batch: [batch_size, 1], y: [1, N] → broadcasts to [batch_size, N]
        x_batch = value_set[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = value_set.unsqueeze(0)  # shape: [1, N]

        # Broadcast add
        z_torch = x_batch + y_full  # shape: [B, N]

        # Send to backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.add(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # ULP check (use bf16 space)
        z_bf16 = z_torch.to(torch.bfloat16).contiguous()
        tt_bf16 = tt_out.to(torch.bfloat16).contiguous()

        z_bits = z_bf16.view(torch.uint16).to(torch.int32)
        tt_bits = tt_bf16.view(torch.uint16).to(torch.int32)

        sign_z = (z_bits >> 15) & 1
        sign_tt = (tt_bits >> 15) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x8000, 0x8000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x8000, 0x8000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        max_ulp_batch = ulp_dist.max().item()
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        # Count ULP distribution
        ulp_0_count += (ulp_dist == 0).sum().item()
        ulp_1_count += (ulp_dist == 1).sum().item()
        ulp_2_count += (ulp_dist == 2).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10)).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100)).sum().item()
        ulp_above_100_count += (ulp_dist > 100).sum().item()

        # Log mismatches to CSV
        mismatch_mask = ulp_dist != 0
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        if mismatch_count > 0:
            # Get mismatch indices in [B, N] grid
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)  # shape: [num_mismatches, 2]

            for idx_pair in mismatch_indices:
                i, j = idx_pair[0].item(), idx_pair[1].item()
                x_idx = start + i
                y_idx = j

                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                zv = z_bf16[i, j].item()
                tv = tt_bf16[i, j].item()
                ulp = ulp_dist[i, j].item()

                xb = z_bf16.new_tensor(xv).view(torch.uint16).item()
                yb = z_bf16.new_tensor(yv).view(torch.uint16).item()
                zb = z_bits[i, j].item() & 0xFFFF
                tb = tt_bits[i, j].item() & 0xFFFF

                writer.writerow([batch_idx, x_idx, y_idx, xv, yv, zv, tv, ulp, xb, yb, zb, tb])

        print(f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}")

    csv_file.close()
    total_pairs = N * N
    mismatch_pct = (total_mismatches / total_pairs) * 100 if total_pairs > 0 else 0.0
    print(f"Global max ULP: {max_ulp_global}, total mismatches: {total_mismatches}/{total_pairs} ({mismatch_pct:.4f}%)")
    print(f"\nULP Distribution:")
    print(f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_pairs*100:.4f}%)")
    print(f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_pairs*100:.4f}%)")
    print(f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_pairs*100:.4f}%)")
    print(f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_pairs*100:.4f}%)")
    print(f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_pairs*100:.4f}%)")
    print(f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_pairs*100:.4f}%)")
    print(f"Mismatch details written to: {out_path_mismatch}")

    assert max_ulp_global <= 2, f"Max ULP {max_ulp_global} exceeds threshold 2"


### -------------------- BF16 FULL RANGE -------------------- ###


def _run_exhaustive_pairwise_add_helper(device, x_values, y_values, test_name, variant_name):
    """Helper function to run exhaustive pairwise addition test between two value sets."""
    ttnn_dtype = ttnn.bfloat16

    # Minimum output magnitude threshold to avoid TT flush-to-zero behavior
    # Values smaller than or equal to this get flushed to zero by TT hardware
    # For bf16, use the same threshold approach as float32 (bf16 has same exponent range)
    finfo_bf16 = np.finfo(np.float32)  # bf16 has same exponent range as float32
    # base_threshold = np.float32(1.175493229783516e-38)  # Same as float32 threshold
    base_threshold = np.float32(torch.finfo(torch.bfloat16).tiny)
    min_output_magnitude = np.nextafter(base_threshold, np.inf)  # Smallest value that won't be flushed

    Nx = x_values.numel()
    Ny = y_values.numel()
    print(f"{test_name}: Testing {Nx} x {Ny} = {Nx*Ny:,} pairs")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_outputs = 0  # Track total valid outputs (after filtering small outputs)

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Open CSV for logging mismatches (variant-specific) - write outside repo
    out_dir = "/home/ubuntu/Repo/Files"
    os.makedirs(out_dir, exist_ok=True)
    out_path_mismatch = os.path.join(out_dir, f"add_ulp_mismatch_exhaustive_{variant_name}.csv")
    csv_file = open(out_path_mismatch, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "batch_idx",
            "x_idx",
            "y_idx",
            "x_value",
            "y_value",
            "torch_out",
            "tt_out",
            "ulp_distance",
            "x_bits_u16",
            "y_bits_u16",
            "torch_bits_u16",
            "tt_bits_u16",
        ]
    )

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch = x_values[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = y_values.unsqueeze(0)  # shape: [1, Ny]

        # Broadcast add
        z_torch = x_batch + y_full  # shape: [B, Ny]

        # Filter out cases where expected output magnitude is too small (will be flushed to zero by TT)
        # Only test cases where abs(output) > min_output_magnitude (strictly greater, since values at threshold also get flushed)
        z_torch_abs = z_torch.abs()
        valid_output_mask = (torch.isfinite(z_torch)) & (z_torch_abs > min_output_magnitude)

        if not valid_output_mask.any():
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"Batch {batch_idx+1}/{num_batches}: All outputs filtered (too small), skipping")
            continue

        # Send to backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.add(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # ULP check (use bf16 space)
        z_bf16 = z_torch.to(torch.bfloat16).contiguous()
        tt_bf16 = tt_out.to(torch.bfloat16).contiguous()

        z_bits = z_bf16.view(torch.uint16).to(torch.int32)
        tt_bits = tt_bf16.view(torch.uint16).to(torch.int32)

        sign_z = (z_bits >> 15) & 1
        sign_tt = (tt_bits >> 15) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x8000, 0x8000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x8000, 0x8000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        # Only consider valid outputs (where expected output magnitude >= min_output_magnitude)
        ulp_dist_valid = ulp_dist[valid_output_mask]
        num_valid_outputs = valid_output_mask.sum().item()
        total_valid_outputs += num_valid_outputs
        num_filtered = valid_output_mask.numel() - num_valid_outputs

        if ulp_dist_valid.numel() > 0:
            max_ulp_batch = ulp_dist_valid.max().item()
            max_ulp_global = max(max_ulp_global, max_ulp_batch)

            # Count ULP distribution (only for valid outputs)
            ulp_0_count += (ulp_dist_valid == 0).sum().item()
            ulp_1_count += (ulp_dist_valid == 1).sum().item()
            ulp_2_count += (ulp_dist_valid == 2).sum().item()
            ulp_3_to_10_count += ((ulp_dist_valid >= 3) & (ulp_dist_valid <= 10)).sum().item()
            ulp_11_to_100_count += ((ulp_dist_valid > 10) & (ulp_dist_valid <= 100)).sum().item()
            ulp_above_100_count += (ulp_dist_valid > 100).sum().item()

            # Log mismatches to CSV (only for valid outputs)
            mismatch_mask = ((ulp_dist > 0) & (ulp_dist <= 2)) & valid_output_mask
            mismatch_count = mismatch_mask.sum().item()
            total_mismatches += mismatch_count
        else:
            max_ulp_batch = 0
            mismatch_count = 0

        if mismatch_count > 0:
            # Get mismatch indices in [B, Ny] grid
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)  # shape: [num_mismatches, 2]

            for idx_pair in mismatch_indices:
                i, j = idx_pair[0].item(), idx_pair[1].item()
                x_idx = start + i
                y_idx = j

                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                zv = z_bf16[i, j].item()
                tv = tt_bf16[i, j].item()
                ulp = ulp_dist[i, j].item()

                xb = z_bf16.new_tensor(xv).view(torch.uint16).item()
                yb = z_bf16.new_tensor(yv).view(torch.uint16).item()
                zb = z_bits[i, j].item() & 0xFFFF
                tb = tt_bits[i, j].item() & 0xFFFF

                writer.writerow([batch_idx, x_idx, y_idx, xv, yv, zv, tv, ulp, xb, yb, zb, tb])

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, valid_outputs={num_valid_outputs}, filtered={num_filtered}"
            )

    csv_file.close()
    total_pairs = Nx * Ny
    total_filtered = total_pairs - total_valid_outputs
    mismatch_pct = (total_mismatches / total_valid_outputs) * 100 if total_valid_outputs > 0 else 0.0

    retained_pct = (total_valid_outputs / total_pairs) * 100 if total_pairs > 0 else 0.0
    filtered_pct = (total_filtered / total_pairs) * 100 if total_pairs > 0 else 0.0

    print("\nOutput filtering summary:")
    print(f"  Total theoretical results: {total_pairs:,}")
    print(f"  Results after filtering (valid outputs): {total_valid_outputs:,} ({retained_pct:.4f}%)")
    print(f"  Filtered out (|output| <= {min_output_magnitude:.9e}): {total_filtered:,} ({filtered_pct:.4f}%)")

    print(
        f"Global max ULP: {max_ulp_global}, total mismatches: {total_mismatches}/{total_valid_outputs} ({mismatch_pct:.4f}%)"
    )
    print(
        f"Total pairs: {total_pairs:,}, Valid outputs: {total_valid_outputs:,}, Filtered (output < {min_output_magnitude}): {total_filtered:,}"
    )
    print(f"\nULP Distribution (valid outputs only):")
    if total_valid_outputs > 0:
        print(f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_outputs*100:.4f}%)")
    else:
        print("  No valid outputs to analyze")

    print(f"Mismatch details written to: {out_path_mismatch}")

    assert max_ulp_global <= 1, f"Max ULP {max_ulp_global} exceeds threshold 1"


def _run_exhaustive_pairwise_add_helper_unique_ulp(device, x_values, y_values, test_name, variant_name):
    """Helper function to run exhaustive pairwise addition test, logging only one example per unique ULP distance."""
    ttnn_dtype = ttnn.bfloat16

    Nx = x_values.numel()
    Ny = y_values.numel()
    print(f"{test_name}: Testing {Nx} x {Ny} = {Nx*Ny:,} pairs")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    logged_ulp_values = set()  # Track which ULP distances we've already logged

    # Open CSV for logging mismatches (variant-specific) - write outside repo
    out_dir = "/home/ubuntu/Repo/Files"
    os.makedirs(out_dir, exist_ok=True)
    out_path_mismatch = os.path.join(out_dir, f"add_ulp_mismatch_exhaustive_{variant_name}_unique.csv")
    csv_file = open(out_path_mismatch, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "batch_idx",
            "x_idx",
            "y_idx",
            "x_value",
            "y_value",
            "torch_out",
            "tt_out",
            "ulp_distance",
            "x_bits_u16",
            "y_bits_u16",
            "torch_bits_u16",
            "tt_bits_u16",
        ]
    )

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch = x_values[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = y_values.unsqueeze(0)  # shape: [1, Ny]

        # Broadcast add
        z_torch = x_batch + y_full  # shape: [B, Ny]

        # Send to backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.add(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # ULP check (use bf16 space)
        z_bf16 = z_torch.to(torch.bfloat16).contiguous()
        tt_bf16 = tt_out.to(torch.bfloat16).contiguous()

        z_bits = z_bf16.view(torch.uint16).to(torch.int32)
        tt_bits = tt_bf16.view(torch.uint16).to(torch.int32)

        sign_z = (z_bits >> 15) & 1
        sign_tt = (tt_bits >> 15) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x8000, 0x8000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x8000, 0x8000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        max_ulp_batch = ulp_dist.max().item()
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        # Log mismatches to CSV - only one example per unique ULP distance
        mismatch_mask = ulp_dist > 0
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        if mismatch_count > 0:
            # Get mismatch indices in [B, Ny] grid
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)  # shape: [num_mismatches, 2]

            # Log only one example per unique ULP distance
            for idx_pair in mismatch_indices:
                i, j = idx_pair[0].item(), idx_pair[1].item()
                ulp = ulp_dist[i, j].item()

                # Skip if we've already logged an example for this ULP distance
                if ulp in logged_ulp_values:
                    continue

                logged_ulp_values.add(ulp)

                x_idx = start + i
                y_idx = j

                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                zv = z_bf16[i, j].item()
                tv = tt_bf16[i, j].item()

                xb = z_bf16.new_tensor(xv).view(torch.uint16).item()
                yb = z_bf16.new_tensor(yv).view(torch.uint16).item()
                zb = z_bits[i, j].item() & 0xFFFF
                tb = tt_bits[i, j].item() & 0xFFFF

                writer.writerow([batch_idx, x_idx, y_idx, xv, yv, zv, tv, ulp, xb, yb, zb, tb])

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}")

    csv_file.close()
    total_pairs = Nx * Ny
    mismatch_pct = (total_mismatches / total_pairs) * 100 if total_pairs > 0 else 0.0
    print(f"Global max ULP: {max_ulp_global}, total mismatches: {total_mismatches}/{total_pairs} ({mismatch_pct:.4f}%)")
    print(f"Unique ULP examples logged: {len(logged_ulp_values)}")
    print(f"Mismatch details written to: {out_path_mismatch}")

    assert max_ulp_global <= 0, f"Max ULP {max_ulp_global} exceeds threshold 0"


def test_add_tt_bf16_exhaustive_pos_pos(device):
    """Test positive x positive normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # for [1.17549435e−38 , 3.38953139e38]
    # Keep only finite, non-zero, positive normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    pos_values = vals[mask]  # ~32512 positive normal values

    # _run_exhaustive_pairwise_add_helper_unique_ulp(device, pos_values, pos_values, "Positive x Positive", "pos_pos")

    _run_exhaustive_pairwise_add_helper(
        device=device,
        x_values=pos_values,
        y_values=pos_values,
        test_name="BF16 Positive x Positive (normal range)",
        variant_name="bf16_pos_pos",
    )


def test_add_tt_bf16_exhaustive_pos_neg(device):
    """Test positive x negative normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # for [1.17549435e−38 , 3.38953139e38]
    # Keep only finite, non-zero, normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    pos_values = vals[pos_mask]  # ~32512 positive normal values
    neg_values = vals[neg_mask]  # ~32512 negative normal values

    # _run_exhaustive_pairwise_add_helper_unique_ulp(device, pos_values, neg_values, "Positive x Negative", "pos_neg")

    _run_exhaustive_pairwise_add_helper(
        device=device,
        x_values=pos_values,
        y_values=neg_values,
        test_name="BF16 Positive x Negative (normal range)",
        variant_name="bf16_pos_neg",
    )


def test_add_tt_bf16_exhaustive_neg_pos(device):
    """Test negative x positive normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # for [1.17549435e−38 , 3.38953139e38]
    # Keep only finite, non-zero, normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    pos_values = vals[pos_mask]  # ~32512 positive normal values
    neg_values = vals[neg_mask]  # ~32512 negative normal values

    # _run_exhaustive_pairwise_add_helper_unique_ulp(device, neg_values, pos_values, "Negative x Positive", "neg_pos")

    _run_exhaustive_pairwise_add_helper(
        device=device,
        x_values=neg_values,
        y_values=pos_values,
        test_name="BF16 Negative x Positive (normal range)",
        variant_name="bf16_neg_pos",
    )


def test_add_tt_bf16_exhaustive_neg_neg(device):
    """Test negative x negative normal bf16 values."""
    torch.manual_seed(0)

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # for [1.17549435e−38 , 3.38953139e38]
    # Keep only finite, non-zero, negative normal bf16 values
    tiny = torch.finfo(torch.bfloat16).tiny
    mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)

    neg_values = vals[mask]  # ~32512 negative normal values

    # _run_exhaustive_pairwise_add_helper_unique_ulp(device, neg_values, neg_values, "Negative x Negative", "neg_neg")

    _run_exhaustive_pairwise_add_helper(
        device=device,
        x_values=neg_values,
        y_values=neg_values,
        test_name="BF16 Negative x Negative (normal range)",
        variant_name="bf16_neg_neg",
    )


### -------------------- BF16 [-1e-20, -1e20] U [1e-20, 1e20] -------------------- ###


def test_add_tt_bf16_tensor_b_fixed(device):
    torch.manual_seed(0)
    ttnn_dtype = ttnn.bfloat16

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)

    # Keep only finite values within [1e-20, 1e20]
    min_abs = torch.tensor(1e-20, dtype=torch.float32)
    max_abs = torch.tensor(1e20, dtype=torch.float32)
    vals_f32_abs = vals.to(torch.float32).abs()
    mask = torch.isfinite(vals) & (vals_f32_abs >= min_abs) & (vals_f32_abs <= max_abs)
    x_torch = vals[mask]

    # Define the three cases for y
    y_cases = {
        "zeros": torch.zeros_like(x_torch),
        "ones": torch.ones_like(x_torch),
        "minus_ones": torch.full_like(x_torch, fill_value=-1.0, dtype=torch.bfloat16),
    }

    results_summary = {}

    for case_name, y_torch in y_cases.items():
        # Reference and backend
        z_torch = torch.add(x_torch, y_torch)
        x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.add(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # Absolute difference
        abs_diff = (z_torch - tt_out).abs()

        # ULP distance
        z_bits32 = z_torch.view(torch.uint16).to(torch.int32)
        tt_bits32 = tt_out.view(torch.uint16).to(torch.int32)
        sign_z = (z_bits32 >> 15) & 1
        sign_tt = (tt_bits32 >> 15) & 1
        z_ord = torch.where(sign_z == 0, z_bits32 + 0x8000, 0x8000 - z_bits32)
        tt_ord = torch.where(sign_tt == 0, tt_bits32 + 0x8000, 0x8000 - tt_bits32)
        ulp_dist = (z_ord - tt_ord).abs()

        # # ----- PLOTS -----
        # out_dir = os.path.dirname(__file__)
        # plot_ulp_histogram(ulp_dist, case_name, out_dir)
        # plot_mismatch_scatter(x_torch, ulp_dist, case_name, out_dir)

        ulp_threshold = 0
        mismatch = ulp_dist > ulp_threshold
        if mismatch.any():
            out_dir = os.path.dirname(__file__)
            out_path_mismatch = os.path.join(out_dir, f"add_ulp_abs_mismatch_cases_range_1e20_1e_20_{case_name}.csv")
            idxs = mismatch.nonzero(as_tuple=False).view(-1).tolist()
            with open(out_path_mismatch, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "index",
                        "x_value",
                        "y_value",
                        "torch_out",
                        "tt_out",
                        "abs_difference",
                        "ulp_distance",
                        "x_bits_u16",
                        "y_bits_u16",
                        "torch_bits_u16",
                        "tt_bits_u16",
                    ]
                )
                x_bits = x_torch[mismatch].view(torch.uint16).tolist()
                y_bits = y_torch[mismatch].view(torch.uint16).tolist()
                z_bits = z_torch[mismatch].view(torch.uint16).tolist()
                t_bits = tt_out[mismatch].view(torch.uint16).tolist()
                for i, xv, yv, zv, tv, absd, ulp, xb, yb, zb, tb in zip(
                    idxs,
                    x_torch[mismatch].tolist(),
                    y_torch[mismatch].tolist(),
                    z_torch[mismatch].tolist(),
                    tt_out[mismatch].tolist(),
                    abs_diff[mismatch].tolist(),
                    ulp_dist[mismatch].tolist(),
                    x_bits,
                    y_bits,
                    z_bits,
                    t_bits,
                ):
                    writer.writerow([i, xv, yv, zv, tv, absd, ulp, xb, yb, zb, tb])
            print(f"ULP mismatches saved to {out_path_mismatch}")
        else:
            print(f"No mismatches for case: {case_name}")

        # Compute ULP statistics
        total = ulp_dist.numel()
        ulp0_count = (ulp_dist == 0).sum().item()
        ulp1_count = (ulp_dist == 1).sum().item()
        other_count = total - ulp0_count - ulp1_count

        results_summary[case_name] = {
            "ulp_dist": ulp_dist,
            "abs_diff": abs_diff,
            "total": total,
            "ulp0_count": ulp0_count,
            "ulp1_count": ulp1_count,
            "other_count": other_count,
        }

        print(f"\nCase: {case_name}")
        print(f"Total values tested: {total}")
        print(f"ULP = 0: {ulp0_count} ({100 * ulp0_count / total:.2f}%)")
        print(f"ULP = 1: {ulp1_count} ({100 * ulp1_count / total:.2f}%)")
        print(f"Other ULPs: {other_count} ({100 * other_count / total:.2f}%)")

    assert_with_ulp(z_torch, tt_out, ulp_threshold=0, allow_nonfinite=False)


def test_add_tt_bf16_range_1e20_1e_20(device):
    torch.manual_seed(0)
    ttnn_dtype = ttnn.bfloat16

    # Generate all possible bf16 values
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(
        torch.uint16
    )  # generates integers 0 → 65535 in int32and stores as raw 16-bit values
    # So all_bitpatterns[i] == bit pattern i
    vals = all_bitpatterns.view(
        torch.bfloat16
    )  # reinterprets the underlying bits as bf16 (no rounding/conversion occurs)

    # Keep only finite values within [1e-15, 1e15] by magnitude (exclude zeros automatically)
    min_abs = torch.tensor(1e-20, dtype=torch.float32)
    max_abs = torch.tensor(1e20, dtype=torch.float32)
    vals_f32_abs = vals.to(torch.float32).abs()
    mask = torch.isfinite(vals) & (vals_f32_abs >= min_abs) & (vals_f32_abs <= max_abs)
    x_torch = vals[mask]

    # Shuffle to create y with same multiset of values
    perm = torch.randperm(x_torch.numel(), device=x_torch.device)  # shuffle the same set ofvalues of x to create y
    y_torch = x_torch.reshape(-1)[perm].reshape_as(x_torch)

    # Reference and backend
    z_torch = torch.add(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.add(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt)

    # Absolute difference
    abs_diff = (z_torch - tt_out).abs()

    # All products in this range should remain finite and comfortably within bf16 limits
    # Log ULP>1 mismatches to CSV for debugging
    z_bits32 = z_torch.view(torch.uint16).to(torch.int32)
    tt_bits32 = tt_out.view(torch.uint16).to(torch.int32)
    sign_z = (z_bits32 >> 15) & 1
    sign_tt = (tt_bits32 >> 15) & 1
    z_ord = torch.where(sign_z == 0, z_bits32 + 0x8000, 0x8000 - z_bits32)
    tt_ord = torch.where(sign_tt == 0, tt_bits32 + 0x8000, 0x8000 - tt_bits32)
    ulp_dist = (z_ord - tt_ord).abs()

    ulp_threshold = 0
    mismatch = ulp_dist > ulp_threshold
    if mismatch.any():
        out_dir = os.path.dirname(__file__)
        out_path_mismatch = os.path.join(out_dir, "add_ulp_mismatch_cases_range_1e20_1e_20.csv")
        idxs = mismatch.nonzero(as_tuple=False).view(-1).tolist()
        with open(out_path_mismatch, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "x_value",
                    "y_value",
                    "torch_out",
                    "tt_out",
                    "abs_difference",
                    "ulp_distance",
                    "x_bits_u16",
                    "y_bits_u16",
                    "torch_bits_u16",
                    "tt_bits_u16",
                ]
            )
            x_bits = x_torch[mismatch].view(torch.uint16).tolist()
            y_bits = y_torch[mismatch].view(torch.uint16).tolist()
            z_bits = z_torch[mismatch].view(torch.uint16).tolist()
            t_bits = tt_out[mismatch].view(torch.uint16).tolist()
            for i, xv, yv, zv, tv, absd, ulp, xb, yb, zb, tb in zip(
                idxs,
                x_torch[mismatch].tolist(),
                y_torch[mismatch].tolist(),
                z_torch[mismatch].tolist(),
                tt_out[mismatch].tolist(),
                abs_diff[mismatch].tolist(),
                ulp_dist[mismatch].tolist(),
                x_bits,
                y_bits,
                z_bits,
                t_bits,
            ):
                writer.writerow([i, xv, yv, zv, tv, absd, ulp, xb, yb, zb, tb])
        print(f"ULP mismatches (>0) in range test: count={mismatch.sum().item()} written to: {out_path_mismatch}")

    # Compute ULP counts and percentages
    total = ulp_dist.numel()
    ulp0_count = (ulp_dist == 0).sum().item()
    ulp1_count = (ulp_dist == 1).sum().item()
    other_ulp_count = total - ulp0_count - ulp1_count

    print(f"Total values tested: {total}")
    print(f"ULP = 0: {ulp0_count} ({100 * ulp0_count / total:.2f}%)")
    print(f"ULP = 1: {ulp1_count} ({100 * ulp1_count / total:.2f}%)")
    print(f"Other ULPs: {other_ulp_count} ({100 * other_ulp_count / total:.2f}%)")

    assert_with_ulp(z_torch, tt_out, ulp_threshold=0, allow_nonfinite=False)


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.add,
    ],
)
def test_add_discrete(ttnn_op, device):
    torch_input_tensor_a = torch.tensor([45, 1.17549435082229e-38], dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([3.859375, 1.18467790043809e-38], dtype=torch.bfloat16)
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)
    torch.set_printoptions(precision=10)
    print(output_tensor)
    print(torch_output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=0, allow_nonfinite=False)


### -------------------- FP32 -------------------- ###


def test_add_tt_float32_sampled_pairwise_full_range(device):
    """Test float32 addition with ~25K strategically sampled values across full normal range."""
    torch.manual_seed(0)
    ttnn_dtype = ttnn.float32

    # Get float32 range limits (excluding inf/nan/zero/subnormals)
    finfo = np.finfo(np.float32)
    min_normal = finfo.tiny  # smallest positive normal ~1.175e-38
    max_normal = finfo.max  # largest positive normal ~3.4e38

    value_list = []

    # 1. Include all powers of 2 in the normal range and their neighbors
    for exp in range(-126, 128):  # float32 exponent range for normals
        base = 2.0**exp
        if min_normal <= abs(base) <= max_normal:
            value_list.extend([base, -base])
            # Add neighbors (next representable float32 values)
            value_list.extend(
                [
                    np.nextafter(np.float32(base), np.inf),
                    np.nextafter(np.float32(base), -np.inf),
                    np.nextafter(np.float32(-base), np.inf),
                    np.nextafter(np.float32(-base), -np.inf),
                ]
            )

    # 2. Add logarithmically spaced samples across the entire normal range
    num_log_samples = 8000  # increased for better coverage
    log_min = np.log10(min_normal)
    log_max = np.log10(max_normal)
    log_samples = np.logspace(log_min, log_max, num_log_samples, dtype=np.float32)
    value_list.extend(log_samples)
    value_list.extend(-log_samples)

    # 3. Add linearly spaced samples in key sub-ranges (both ends of spectrum)
    for range_start, range_end, num_samples in [
        # Very small normals
        (min_normal, 1e-30, 300),
        (1e-30, 1e-20, 300),
        (1e-20, 1e-10, 300),
        # Small to medium
        (1e-10, 1e-5, 300),
        (1e-5, 0.1, 300),
        (0.1, 1.0, 300),
        # Medium to large
        (1.0, 100.0, 300),
        (100.0, 1e5, 300),
        (1e5, 1e10, 300),
        # Very large
        (1e10, 1e20, 300),
        (1e20, 1e30, 300),
        (1e30, max_normal, 300),
    ]:
        linear_samples = np.linspace(range_start, range_end, num_samples, dtype=np.float32)
        value_list.extend(linear_samples)
        value_list.extend(-linear_samples)

    # 4. Add random samples across the full range (log-uniform distribution)
    num_random = 3000
    # Generate random exponents uniformly in log space
    random_log_vals = np.random.uniform(log_min, log_max, num_random)
    random_samples = (10**random_log_vals).astype(np.float32)
    value_list.extend(random_samples)
    value_list.extend(-random_samples)

    # 5. Add some special interesting values
    special_values = [
        1.0,
        -1.0,
        np.e,
        -np.e,
        np.pi,
        -np.pi,
        10.0,
        -10.0,
        100.0,
        -100.0,
        0.5,
        -0.5,
        0.1,
        -0.1,
        min_normal,
        -min_normal,  # smallest normals
        max_normal,
        -max_normal,  # largest normals
    ]
    value_list.extend([np.float32(v) for v in special_values])

    # Convert to tensor, remove duplicates, and filter
    vals = torch.tensor(value_list, dtype=torch.float32)
    vals_abs = vals.abs()
    # Filter: finite, non-zero, normal values only
    mask = torch.isfinite(vals) & (vals != 0) & (vals_abs >= min_normal) & (vals_abs <= max_normal)
    vals_filtered = vals[mask]

    # Remove duplicates
    value_set = torch.unique(vals_filtered)
    N = value_set.numel()
    print(f"Testing {N} float32 values (full normal range) → {N}x{N} = {N*N:,} pairs")

    # Batch processing to avoid OOM
    batch_size = 256  # smaller batch for float32 (4 bytes vs 2 bytes for bf16)
    num_batches = (N + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Open CSV for logging mismatches
    out_dir = os.path.dirname(__file__)
    out_path_mismatch = os.path.join(out_dir, "add_ulp_mismatch_float32_25K_BH.csv")
    csv_file = open(out_path_mismatch, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "batch_idx",
            "x_idx",
            "y_idx",
            "x_value",
            "y_value",
            "torch_out",
            "tt_out",
            "ulp_distance",
            "x_bits_u32",
            "y_bits_u32",
            "torch_bits_u32",
            "tt_bits_u32",
        ]
    )

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        # x_batch: [batch_size, 1], y: [1, N] → broadcasts to [batch_size, N]
        x_batch = value_set[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = value_set.unsqueeze(0)  # shape: [1, N]

        # Broadcast add (both inputs are float32)
        z_torch = x_batch + y_full  # shape: [B, N]

        # Send to backend (using float32)
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.add(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # ULP check (use float32 space)
        z_f32 = z_torch.to(torch.float32).contiguous()
        tt_f32 = tt_out.to(torch.float32).contiguous()

        # Convert to uint32 for bitwise comparison
        z_bits = z_f32.view(torch.int32)
        tt_bits = tt_f32.view(torch.int32)

        # Calculate ULP distance using signed magnitude representation
        sign_z = (z_bits >> 31) & 1
        sign_tt = (tt_bits >> 31) & 1

        # Convert to ordered representation for ULP calculation
        z_ord = torch.where(sign_z == 0, z_bits + 0x80000000, 0x80000000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x80000000, 0x80000000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        max_ulp_batch = ulp_dist.max().item()
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        # Count ULP distribution
        ulp_0_count += (ulp_dist == 0).sum().item()
        ulp_1_count += (ulp_dist == 1).sum().item()
        ulp_2_count += (ulp_dist == 2).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10)).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100)).sum().item()
        ulp_above_100_count += (ulp_dist > 100).sum().item()

        # Log mismatches to CSV
        mismatch_mask = ulp_dist > 1
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        if mismatch_count > 0:
            # Get mismatch indices in [B, N] grid
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)  # shape: [num_mismatches, 2]

            for idx_pair in mismatch_indices:
                i, j = idx_pair[0].item(), idx_pair[1].item()
                x_idx = start + i
                y_idx = j

                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                zv = z_f32[i, j].item()
                tv = tt_f32[i, j].item()
                ulp = ulp_dist[i, j].item()

                # Get bit representations as uint32
                xb = z_f32.new_tensor(xv).view(torch.int32).item() & 0xFFFFFFFF
                yb = z_f32.new_tensor(yv).view(torch.int32).item() & 0xFFFFFFFF
                zb = z_bits[i, j].item() & 0xFFFFFFFF
                tb = tt_bits[i, j].item() & 0xFFFFFFFF

                writer.writerow([batch_idx, x_idx, y_idx, xv, yv, zv, tv, ulp, xb, yb, zb, tb])

        print(f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}")

    csv_file.close()
    total_pairs = N * N
    mismatch_pct = (total_mismatches / total_pairs) * 100 if total_pairs > 0 else 0.0
    print(f"Global max ULP: {max_ulp_global}, total mismatches: {total_mismatches}/{total_pairs} ({mismatch_pct:.4f}%)")
    print(f"\nULP Distribution:")
    print(f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_pairs*100:.4f}%)")
    print(f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_pairs*100:.4f}%)")
    print(f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_pairs*100:.4f}%)")
    print(f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_pairs*100:.4f}%)")
    print(f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_pairs*100:.4f}%)")
    print(f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_pairs*100:.4f}%)")
    print(f"Mismatch details written to: {out_path_mismatch}")

    assert max_ulp_global <= 2, f"Max ULP {max_ulp_global} exceeds threshold 2"


### -------------------- FP32 OUTPUTS FILTERED TO AVOID TT FLUSH-TO-ZERO BEHAVIOR -------------------- ###


def test_add_tt_float32_sampled_pairwise_full_range_valid_output(device):
    """Test float32 addition with ~25K strategically sampled values across full normal range."""
    torch.manual_seed(0)
    np.random.seed(0)
    ttnn_dtype = ttnn.float32

    # Get float32 range limits (excluding inf/nan/zero/subnormals)
    finfo = np.finfo(np.float32)
    min_normal = finfo.tiny  # smallest positive normal ~1.175e-38
    max_normal = finfo.max  # largest positive normal ~3.4e38

    # Minimum output magnitude threshold to avoid TT flush-to-zero behavior
    # Values smaller than or equal to this get flushed to zero by TT hardware
    # Add small epsilon to be safe (next representable float32 value)
    base_threshold = np.float32(torch.finfo(torch.bfloat16).tiny)
    min_output_magnitude = np.nextafter(base_threshold, np.inf)  # Smallest value that won't be flushed

    value_list = []

    # 1. Include all powers of 2 in the normal range and their neighbors
    for exp in range(-126, 128):  # float32 exponent range for normals
        base = 2.0**exp
        if min_normal <= abs(base) <= max_normal:
            value_list.extend([base, -base])
            # Add neighbors (next representable float32 values)
            value_list.extend(
                [
                    np.nextafter(np.float32(base), np.inf),
                    np.nextafter(np.float32(base), -np.inf),
                    np.nextafter(np.float32(-base), np.inf),
                    np.nextafter(np.float32(-base), -np.inf),
                ]
            )

    # 2. Add logarithmically spaced samples across the entire normal range
    num_log_samples = 8000  # increased for better coverage
    log_min = np.log10(min_normal)
    log_max = np.log10(max_normal)
    log_samples = np.logspace(log_min, log_max, num_log_samples, dtype=np.float32)
    value_list.extend(log_samples)
    value_list.extend(-log_samples)

    # 3. Add linearly spaced samples in key sub-ranges (both ends of spectrum)
    for range_start, range_end, num_samples in [
        # Very small normals
        (min_normal, 1e-30, 300),
        (1e-30, 1e-20, 300),
        (1e-20, 1e-10, 300),
        # Small to medium
        (1e-10, 1e-5, 300),
        (1e-5, 0.1, 300),
        (0.1, 1.0, 300),
        # Medium to large
        (1.0, 100.0, 300),
        (100.0, 1e5, 300),
        (1e5, 1e10, 300),
        # Very large
        (1e10, 1e20, 300),
        (1e20, 1e30, 300),
        (1e30, max_normal, 300),
    ]:
        linear_samples = np.linspace(range_start, range_end, num_samples, dtype=np.float32)
        value_list.extend(linear_samples)
        value_list.extend(-linear_samples)

    # 4. Add random samples across the full range (log-uniform distribution)
    num_random = 3000
    # Generate random exponents uniformly in log space
    random_log_vals = np.random.uniform(log_min, log_max, num_random)
    random_samples = (10**random_log_vals).astype(np.float32)
    value_list.extend(random_samples)
    value_list.extend(-random_samples)

    # 5. Add some special interesting values
    special_values = [
        1.0,
        -1.0,
        np.e,
        -np.e,
        np.pi,
        -np.pi,
        10.0,
        -10.0,
        100.0,
        -100.0,
        0.5,
        -0.5,
        0.1,
        -0.1,
        min_normal,
        -min_normal,  # smallest normals
        max_normal,
        -max_normal,  # largest normals
    ]
    value_list.extend([np.float32(v) for v in special_values])

    # Convert to tensor, remove duplicates, and filter
    vals = torch.tensor(value_list, dtype=torch.float32)
    vals_abs = vals.abs()
    # Filter: finite, non-zero, normal values only
    mask = torch.isfinite(vals) & (vals != 0) & (vals_abs >= min_normal) & (vals_abs <= max_normal)
    vals_filtered = vals[mask]

    # Remove duplicates
    value_set = torch.unique(vals_filtered)
    N = value_set.numel()
    print(f"Testing {N} float32 values (full normal range) → {N}x{N} = {N*N:,} pairs")

    # Batch processing to avoid OOM
    batch_size = 256  # smaller batch for float32 (4 bytes vs 2 bytes for bf16)
    num_batches = (N + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_outputs = 0  # Track total valid outputs (after filtering small outputs)

    # Track ULP distribution
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0

    # Open CSV for logging mismatches
    out_dir = os.path.dirname(__file__)
    out_path_mismatch = os.path.join(out_dir, "add_ulp_mismatch_float32_25K_BH.csv")
    csv_file = open(out_path_mismatch, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "batch_idx",
            "x_idx",
            "y_idx",
            "x_value",
            "y_value",
            "torch_out",
            "tt_out",
            "ulp_distance",
            "x_bits_u32",
            "y_bits_u32",
            "torch_bits_u32",
            "tt_bits_u32",
        ]
    )

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        # x_batch: [batch_size, 1], y: [1, N] → broadcasts to [batch_size, N]
        x_batch = value_set[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = value_set.unsqueeze(0)  # shape: [1, N]

        # Broadcast add (both inputs are float32)
        z_torch = x_batch + y_full  # shape: [B, N]

        # Filter out cases where expected output magnitude is too small (will be flushed to zero by TT)
        # Only test cases where abs(output) > min_output_magnitude (strictly greater, since values at threshold also get flushed)
        z_torch_abs = z_torch.abs()
        valid_output_mask = (torch.isfinite(z_torch)) & (z_torch_abs > min_output_magnitude)

        if not valid_output_mask.any():
            print(f"Batch {batch_idx+1}/{num_batches}: All outputs filtered (too small), skipping")
            continue

        # Send to backend (using float32)
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.add(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # ULP check (use float32 space)
        z_f32 = z_torch.to(torch.float32).contiguous()
        tt_f32 = tt_out.to(torch.float32).contiguous()

        # Convert to uint32 for bitwise comparison
        z_bits = z_f32.view(torch.int32)
        tt_bits = tt_f32.view(torch.int32)

        # Calculate ULP distance using signed magnitude representation
        sign_z = (z_bits >> 31) & 1
        sign_tt = (tt_bits >> 31) & 1

        # Convert to ordered representation for ULP calculation
        z_ord = torch.where(sign_z == 0, z_bits + 0x80000000, 0x80000000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x80000000, 0x80000000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()

        # Only consider valid outputs (where expected output magnitude >= min_output_magnitude)
        ulp_dist_valid = ulp_dist[valid_output_mask]

        if ulp_dist_valid.numel() > 0:
            max_ulp_batch = ulp_dist_valid.max().item()
            max_ulp_global = max(max_ulp_global, max_ulp_batch)

            # Count ULP distribution (only for valid outputs)
            ulp_0_count += (ulp_dist_valid == 0).sum().item()
            ulp_1_count += (ulp_dist_valid == 1).sum().item()
            ulp_2_count += (ulp_dist_valid == 2).sum().item()
            ulp_3_to_10_count += ((ulp_dist_valid >= 3) & (ulp_dist_valid <= 10)).sum().item()
            ulp_11_to_100_count += ((ulp_dist_valid >= 11) & (ulp_dist_valid <= 100)).sum().item()
            ulp_above_100_count += (ulp_dist_valid > 100).sum().item()

            # Log mismatches to CSV (only for valid outputs)
            mismatch_mask = (ulp_dist > 1) & valid_output_mask
            mismatch_count = mismatch_mask.sum().item()
            total_mismatches += mismatch_count
        else:
            max_ulp_batch = 0
            mismatch_count = 0

        if mismatch_count > 0:
            # Get mismatch indices in [B, N] grid (only for valid outputs)
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)  # shape: [num_mismatches, 2]

            for idx_pair in mismatch_indices:
                i, j = idx_pair[0].item(), idx_pair[1].item()
                x_idx = start + i
                y_idx = j

                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                zv = z_f32[i, j].item()
                tv = tt_f32[i, j].item()
                ulp = ulp_dist[i, j].item()

                # Get bit representations as uint32
                xb = z_f32.new_tensor(xv).view(torch.int32).item() & 0xFFFFFFFF
                yb = z_f32.new_tensor(yv).view(torch.int32).item() & 0xFFFFFFFF
                zb = z_bits[i, j].item() & 0xFFFFFFFF
                tb = tt_bits[i, j].item() & 0xFFFFFFFF

                writer.writerow([batch_idx, x_idx, y_idx, xv, yv, zv, tv, ulp, xb, yb, zb, tb])

        num_valid_outputs = valid_output_mask.sum().item()
        total_valid_outputs += num_valid_outputs
        num_filtered = valid_output_mask.numel() - num_valid_outputs
        print(
            f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, valid_outputs={num_valid_outputs}, filtered={num_filtered}"
        )

    csv_file.close()
    total_pairs = N * N
    total_filtered = total_pairs - total_valid_outputs

    retained_pct = (total_valid_outputs / total_pairs) * 100 if total_pairs > 0 else 0.0
    filtered_pct = (total_filtered / total_pairs) * 100 if total_pairs > 0 else 0.0

    print("\nOutput filtering summary:")
    print(f"  Total theoretical results: {total_pairs:,}")
    print(f"  Results after filtering (valid outputs): {total_valid_outputs:,} ({retained_pct:.4f}%)")
    print(f"  Filtered out (|output| <= {min_output_magnitude:.9e}): {total_filtered:,} ({filtered_pct:.4f}%)")

    mismatch_pct = (total_mismatches / total_valid_outputs) * 100 if total_valid_outputs > 0 else 0.0
    print(
        f"Global max ULP: {max_ulp_global}, total mismatches: {total_mismatches}/{total_valid_outputs} ({mismatch_pct:.4f}%)"
    )
    print(
        f"Total pairs: {total_pairs:,}, Valid outputs: {total_valid_outputs:,}, Filtered (output < {min_output_magnitude}): {total_filtered:,}"
    )
    print(f"\nULP Distribution (valid outputs only):")
    if total_valid_outputs > 0:
        print(f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_outputs*100:.4f}%)")
        print(f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_outputs*100:.4f}%)")
    else:
        print("  No valid outputs to analyze")
    print(f"Mismatch details written to: {out_path_mismatch}")

    assert max_ulp_global <= 2, f"Max ULP {max_ulp_global} exceeds threshold 2"
