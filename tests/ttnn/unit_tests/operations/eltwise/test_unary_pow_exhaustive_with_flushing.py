# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import os
import csv
import pytest
import ttnn
import numpy as np
import matplotlib.pyplot as plt
from tests.ttnn.utils_for_testing import assert_with_ulp, flush_subnormal_values_to_zero

# BF16
# pytest /home/ubuntu/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_unary_pow_accuracy.py::test_pow_tt_BF16_exhaustive_test

# FP32
# pytest /home/ubuntu/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_unary_pow_accuracy.py::test_pow_tt_FP32_exhaustive_test


def _run_exhaustive_unary_pow_helper_exhaustive_exponents(
    device,
    base_values,
    exponent_values,
    test_name,
    variant_name,
    max_skipped_samples=10,
    progress_every=10,
):
    """Exhaustive unary pow: sweep over all base_values and all exponent_values.

    Uses unary ttnn.pow(base_tensor, exponent_scalar): one device launch per (batch, exponent).
    Pairs where the torch (reference) result is NaN or Inf are SKIPPED from accuracy
    comparison.
    """
    ttnn_dtype = ttnn.bfloat16
    N_base = base_values.numel()
    N_exp = exponent_values.numel()
    total_pairs = N_base * N_exp
    print(f"\n{'='*60}")
    print(f"{test_name}: Testing {N_base:,} bases × {N_exp:,} exponents = {total_pairs:,} pairs (unary pow)")

    batch_size = 512
    num_batches = (N_base + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_pairs = 0
    total_skipped_pairs = 0
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0
    skipped_samples = []

    # Flush exponents to scalars (subnormal/non-finite -> 0.0)
    exp_flushed = exponent_values.clone()
    flush_subnormal_values_to_zero(exp_flushed)
    exp_flushed[~torch.isfinite(exp_flushed)] = 0.0
    exp_scalars = [exp_flushed[i].item() for i in range(N_exp)]

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N_base)
        x_batch = base_values[start:end].unsqueeze(1).clone()  # [B, 1]
        flush_subnormal_values_to_zero(x_batch)
        x_batch[~torch.isfinite(x_batch)] = 0.0

        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        max_ulp_batch = 0
        mismatch_count_batch = 0
        valid_count_batch = 0
        total_batch_pairs = 0

        for exp_idx, exp_scalar in enumerate(exp_scalars):
            z_torch = torch.pow(x_batch, exp_scalar)  # [B, 1]
            z_tt = ttnn.pow(x_tt, exp_scalar)
            tt_out = ttnn.to_torch(z_tt)

            z_bf16 = z_torch.to(torch.bfloat16).contiguous()
            tt_bf16 = tt_out.to(torch.bfloat16).contiguous()
            valid_mask = torch.isfinite(z_bf16)
            skipped_mask = ~valid_mask
            total_skipped_pairs += skipped_mask.sum().item()

            if skipped_mask.any() and len(skipped_samples) < max_skipped_samples:
                skipped_indices = skipped_mask.nonzero(as_tuple=False)
                for idx_row in skipped_indices:
                    if len(skipped_samples) >= max_skipped_samples:
                        break
                    i = idx_row[0].item()
                    reason = "NaN" if torch.isnan(z_bf16[i, 0]) else ("+Inf" if z_bf16[i, 0] > 0 else "-Inf")
                    skipped_samples.append(
                        {
                            "base": x_batch[i, 0].item(),
                            "exponent": exp_scalar,
                            "torch_result": z_bf16[i, 0].item(),
                            "ttnn_result": tt_bf16[i, 0].item(),
                            "skip_reason": reason,
                            "category": variant_name,
                        }
                    )

            flush_subnormal_values_to_zero(z_bf16)
            flush_subnormal_values_to_zero(tt_bf16)
            z_bf16[~torch.isfinite(z_bf16)] = 0.0
            tt_bf16[~torch.isfinite(tt_bf16)] = 0.0

            z_bits = z_bf16.view(torch.uint16).to(torch.int32)
            tt_bits = tt_bf16.view(torch.uint16).to(torch.int32)
            sign_z = (z_bits >> 15) & 1
            sign_tt = (tt_bits >> 15) & 1
            z_ord = torch.where(sign_z == 0, z_bits + 0x8000, 0x8000 - z_bits)
            tt_ord = torch.where(sign_tt == 0, tt_bits + 0x8000, 0x8000 - tt_bits)
            ulp_dist = (z_ord - tt_ord).abs()
            valid_count = valid_mask.sum().item()
            total_batch_pairs += x_batch.shape[0]
            max_ulp_batch = max(max_ulp_batch, ulp_dist[valid_mask].max().item() if valid_count > 0 else 0)
            max_ulp_global = max(max_ulp_global, max_ulp_batch)
            ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
            ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
            ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
            ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
            ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
            ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()
            mismatch_count_batch += ((ulp_dist > 1) & valid_mask).sum().item()
            total_mismatches += ((ulp_dist > 1) & valid_mask).sum().item()
            total_valid_pairs += valid_count
            valid_count_batch += valid_count

        if progress_every and ((batch_idx + 1) % progress_every == 0 or batch_idx == num_batches - 1):
            print(
                f"  Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count_batch}, tested={valid_count_batch}/{total_batch_pairs}"
            )

    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    skipped_pct = (total_skipped_pairs / total_pairs) * 100 if total_pairs > 0 else 0.0
    print(f"\n{'='*60}")
    # print(f"Valid (torch finite): pairs used for ULP; skipped (torch NaN/Inf) excluded")
    print(f"Total input pairs: {total_pairs:,}")
    print(f"Valid pairs (torch finite): {total_valid_pairs:,} ({total_valid_pairs/total_pairs*100:.4f}%)")
    print(f"Skipped pairs (torch NaN/Inf): {total_skipped_pairs:,} ({skipped_pct:.4f}%)")
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
    return skipped_samples


def _run_exhaustive_unary_pow_helper_float32_exhaustive_exponents(
    device,
    base_values,
    exponent_values,
    test_name,
    variant_name,
    max_skipped_samples=10,
    progress_every=10,
):
    """Exhaustive unary pow in float32: all base_values × all exponent_values.

    Uses unary ttnn.pow(base_tensor, exponent_scalar): one device launch per (batch, exponent).
    Pairs where the torch (reference) result is NaN or Inf are SKIPPED from accuracy
    comparison.
    """
    ttnn_dtype = ttnn.float32
    N_base = base_values.numel()
    N_exp = exponent_values.numel()
    total_pairs = N_base * N_exp
    print(f"\n{'='*60}")
    print(f"{test_name} (float32): Testing {N_base:,} bases × {N_exp:,} exponents = {total_pairs:,} pairs (unary pow)")

    batch_size = 512
    num_batches = (N_base + batch_size - 1) // batch_size

    max_ulp_global = 0
    total_mismatches = 0
    total_valid_pairs = 0
    total_skipped_pairs = 0
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0
    skipped_samples = []

    # Flush exponents to scalars (subnormal/non-finite -> 0.0)
    exp_flushed = exponent_values.to(torch.float32).clone()
    flush_subnormal_values_to_zero(exp_flushed)
    exp_flushed[~torch.isfinite(exp_flushed)] = 0.0
    exp_scalars = [exp_flushed[i].item() for i in range(N_exp)]

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N_base)
        x_batch_f32 = base_values[start:end].to(torch.float32).unsqueeze(1).clone()  # [B, 1]
        flush_subnormal_values_to_zero(x_batch_f32)
        x_batch_f32[~torch.isfinite(x_batch_f32)] = 0.0

        x_tt = ttnn.from_torch(x_batch_f32, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        max_ulp_batch = 0
        mismatch_count_batch = 0
        valid_count_batch = 0
        total_batch_pairs = 0

        for exp_idx, exp_scalar in enumerate(exp_scalars):
            z_torch = torch.pow(x_batch_f32, exp_scalar)  # [B, 1]
            z_tt = ttnn.pow(x_tt, exp_scalar)
            tt_out = ttnn.to_torch(z_tt)

            z_f32 = z_torch.to(torch.float32).contiguous()
            tt_f32 = tt_out.to(torch.float32).contiguous()
            valid_mask = torch.isfinite(z_f32)
            skipped_mask = ~valid_mask
            total_skipped_pairs += skipped_mask.sum().item()

            if skipped_mask.any() and len(skipped_samples) < max_skipped_samples:
                skipped_indices = skipped_mask.nonzero(as_tuple=False)
                for idx_row in skipped_indices:
                    if len(skipped_samples) >= max_skipped_samples:
                        break
                    i = idx_row[0].item()
                    reason = "NaN" if torch.isnan(z_f32[i, 0]) else ("+Inf" if z_f32[i, 0] > 0 else "-Inf")
                    skipped_samples.append(
                        {
                            "base": x_batch_f32[i, 0].item(),
                            "exponent": exp_scalar,
                            "torch_result": z_f32[i, 0].item(),
                            "ttnn_result": tt_f32[i, 0].item(),
                            "skip_reason": reason,
                            "category": variant_name,
                        }
                    )

            flush_subnormal_values_to_zero(z_f32)
            flush_subnormal_values_to_zero(tt_f32)
            z_f32[~torch.isfinite(z_f32)] = 0.0
            tt_f32[~torch.isfinite(tt_f32)] = 0.0

            z_bits = z_f32.view(torch.int32)
            tt_bits = tt_f32.view(torch.int32)
            sign_z = (z_bits >> 31) & 1
            sign_tt = (tt_bits >> 31) & 1
            z_ord = torch.where(sign_z == 0, z_bits + 0x80000000, 0x80000000 - z_bits)
            tt_ord = torch.where(sign_tt == 0, tt_bits + 0x80000000, 0x80000000 - tt_bits)
            ulp_dist = (z_ord - tt_ord).abs()
            valid_count = valid_mask.sum().item()
            total_batch_pairs += x_batch_f32.shape[0]
            max_ulp_batch = max(max_ulp_batch, ulp_dist[valid_mask].max().item() if valid_count > 0 else 0)
            max_ulp_global = max(max_ulp_global, max_ulp_batch)
            ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
            ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
            ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
            ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
            ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
            ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()
            mismatch_count_batch += ((ulp_dist > 1) & valid_mask).sum().item()
            total_mismatches += ((ulp_dist > 1) & valid_mask).sum().item()
            total_valid_pairs += valid_count
            valid_count_batch += valid_count

        if progress_every and ((batch_idx + 1) % progress_every == 0 or batch_idx == num_batches - 1):
            print(
                f"  Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count_batch}, tested={valid_count_batch}/{total_batch_pairs}"
            )

    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    skipped_pct = (total_skipped_pairs / total_pairs) * 100 if total_pairs > 0 else 0.0
    print(f"\n{'='*60}")
    # print(f"Valid (torch finite): pairs used for ULP; skipped (torch NaN/Inf) excluded")
    print(f"Total input pairs: {total_pairs:,}")
    print(f"Valid pairs (torch finite): {total_valid_pairs:,} ({total_valid_pairs/total_pairs*100:.4f}%)")
    print(f"Skipped pairs (torch NaN/Inf): {total_skipped_pairs:,} ({skipped_pct:.4f}%)")
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
    return skipped_samples


def _generate_normal_finite_nonzero_bf16_values():
    """All normal finite non-zero bf16 values, split into positive and negative."""
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)
    return vals[pos_mask], vals[neg_mask]


# Specific exponent lists (same as test_fill.py): used for positive_base and negative_base categories.
EXPONENT_FP32 = [
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    -7.25,
    -15.875,
    -0.75,
    -0.3125,
    0.125,
    0.6000000238418579,
    1.2999999523162842,
    5.75,
    12.399999618530273,
    1.2345000505447388,
    2.718280076980591,
    3.141590118408203,
    10.123000144958496,
]
EXPONENT_BF16 = [
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    -7.25,
    -15.875,
    -0.75,
    -0.3125,
    0.125,
    0.59765625,
    1.296875,
    5.75,
    12.375,
    1.234375,
    2.703125,
    3.140625,
    10.0625,
]


# FP32 exhaustive tests: two categories (positive base, negative base), each with EXPONENT_FP32.
def test_pow_tt_FP32_exhaustive_positive_base(device):
    """FP32: positive base ^ exponents (EXPONENT_FP32)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    exponent_values = torch.tensor(EXPONENT_FP32, dtype=torch.float32)
    return _run_exhaustive_unary_pow_helper_float32_exhaustive_exponents(
        device,
        pos_values,
        exponent_values,
        "Positive base ^ exponent (FP32 exponents)",
        "positive_base",
    )


def test_pow_tt_FP32_exhaustive_negative_base(device):
    """FP32: negative base ^ exponents (EXPONENT_FP32)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    exponent_values = torch.tensor(EXPONENT_FP32, dtype=torch.float32)
    return _run_exhaustive_unary_pow_helper_float32_exhaustive_exponents(
        device,
        neg_values,
        exponent_values,
        "Negative base ^ exponent (FP32 exponents)",
        "negative_base",
    )


@pytest.mark.timeout(1800)
def test_pow_tt_FP32_exhaustive_test(device):
    """Unary pow (base**exponent) with fp32 - positive_base and negative_base categories, EXPONENT_FP32."""
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

    # Collect skipped samples from both categories (positive_base, negative_base)
    all_skipped_samples = []

    try:
        all_skipped_samples.extend(test_pow_tt_FP32_exhaustive_positive_base(device))
        all_skipped_samples.extend(test_pow_tt_FP32_exhaustive_negative_base(device))
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
        writer.writerow(["category", "base", "exponent", "torch_result", "ttnn_result", "skip_reason"])
        for sample in all_skipped_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["base"],
                    sample["exponent"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["skip_reason"],
                ]
            )

    print(f"\n{'='*60}")
    print(f"All results saved to: {out_path}")
    print(f"Skipped samples CSV ({len(all_skipped_samples)} rows): {skipped_csv_path}")
    print(f"{'='*60}")


# BF16 Exhaustive tests: two categories (positive base, negative base), each with EXPONENT_BF16.
def test_pow_tt_bf16_exhaustive_positive_base(device):
    """BF16: positive base ^ exponents (EXPONENT_BF16)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    exponent_values = torch.tensor(EXPONENT_BF16, dtype=torch.bfloat16)
    return _run_exhaustive_unary_pow_helper_exhaustive_exponents(
        device,
        pos_values,
        exponent_values,
        "Positive base ^ exponent (BF16 exponents)",
        "positive_base",
    )


def test_pow_tt_bf16_exhaustive_negative_base(device):
    """BF16: negative base ^ exponents (EXPONENT_BF16)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    exponent_values = torch.tensor(EXPONENT_BF16, dtype=torch.bfloat16)
    return _run_exhaustive_unary_pow_helper_exhaustive_exponents(
        device,
        neg_values,
        exponent_values,
        "Negative base ^ exponent (BF16 exponents)",
        "negative_base",
    )


@pytest.mark.timeout(1800)
def test_pow_tt_BF16_exhaustive_test(device):
    """Unary pow (base**exponent) with BF16 - positive_base and negative_base categories, EXPONENT_BF16."""
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

    # Collect skipped samples from both categories (positive_base, negative_base)
    all_skipped_samples = []

    try:
        all_skipped_samples.extend(test_pow_tt_bf16_exhaustive_positive_base(device))
        all_skipped_samples.extend(test_pow_tt_bf16_exhaustive_negative_base(device))
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
        writer.writerow(["category", "base", "exponent", "torch_result", "ttnn_result", "skip_reason"])
        for sample in all_skipped_samples:
            writer.writerow(
                [
                    sample["category"],
                    sample["base"],
                    sample["exponent"],
                    sample["torch_result"],
                    sample["ttnn_result"],
                    sample["skip_reason"],
                ]
            )

    print(f"\n{'='*60}")
    print(f"All results saved to: {out_path}")
    print(f"Skipped samples CSV ({len(all_skipped_samples)} rows): {skipped_csv_path}")
    print(f"{'='*60}")
