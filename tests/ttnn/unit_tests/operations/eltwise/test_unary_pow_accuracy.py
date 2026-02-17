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

    Uses binary ttnn.pow(base_tensor, exponent_tensor) with shapes [B, 1] and [1, N_exp]
    so each device launch covers B×N_exp pairs (same batching strategy as fmod test).
    Pairs where the torch (reference) result is NaN or Inf are SKIPPED from accuracy
    comparison.
    """
    ttnn_dtype = ttnn.bfloat16
    N_base = base_values.numel()
    N_exp = exponent_values.numel()
    total_pairs = N_base * N_exp
    print(f"\n{'='*60}")
    print(f"{test_name}: Testing {N_base:,} bases × {N_exp:,} exponents = {total_pairs:,} pairs")

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

    # Broadcast: base [B, 1], exponent [1, N_exp] -> one kernel per base batch
    exp_full = exponent_values.unsqueeze(0).clone()  # [1, N_exp]
    # Flush input exponents: subnormal and non-finite to zero (once, reused for all batches)
    flush_subnormal_values_to_zero(exp_full)
    exp_full[~torch.isfinite(exp_full)] = 0.0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N_base)
        x_batch = base_values[start:end].unsqueeze(1).clone()  # [B, 1]
        # Flush input bases: subnormal and non-finite to zero
        flush_subnormal_values_to_zero(x_batch)
        x_batch[~torch.isfinite(x_batch)] = 0.0
        z_torch = torch.pow(x_batch, exp_full)  # [B, N_exp]

        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        exp_tt = ttnn.from_torch(exp_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.pow(x_tt, exp_tt)
        tt_out = ttnn.to_torch(z_tt)

        z_bf16 = z_torch.to(torch.bfloat16).contiguous()
        tt_bf16 = tt_out.to(torch.bfloat16).contiguous()
        valid_mask = torch.isfinite(z_bf16)
        skipped_mask = ~valid_mask
        total_skipped_pairs += skipped_mask.sum().item()

        if skipped_mask.any() and len(skipped_samples) < max_skipped_samples:
            skipped_indices = skipped_mask.nonzero(as_tuple=False)
            for idx_pair in skipped_indices:
                if len(skipped_samples) >= max_skipped_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                reason = "NaN" if torch.isnan(z_bf16[i, j]) else ("+Inf" if z_bf16[i, j] > 0 else "-Inf")
                skipped_samples.append(
                    {
                        "base": x_batch[i, 0].item(),
                        "exponent": exp_full[0, j].item(),
                        "torch_result": z_bf16[i, j].item(),
                        "ttnn_result": tt_bf16[i, j].item(),
                        "skip_reason": reason,
                        "category": variant_name,
                    }
                )

        # Flush subnormal and non-finite (outputs) to zero
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
        max_ulp_batch = ulp_dist[valid_mask].max().item() if valid_count > 0 else 0
        max_ulp_global = max(max_ulp_global, max_ulp_batch)
        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()
        mismatch_count = ((ulp_dist > 1) & valid_mask).sum().item()
        total_mismatches += mismatch_count
        total_valid_pairs += valid_count
        total_batch_pairs = x_batch.shape[0] * exp_full.shape[1]

        if progress_every and ((batch_idx + 1) % progress_every == 0 or batch_idx == num_batches - 1):
            print(
                f"  Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, tested={valid_count}/{total_batch_pairs}"
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

    Uses binary ttnn.pow(base_tensor, exponent_tensor) with shapes [B, 1] and [1, N_exp]
    so each device launch covers B×N_exp pairs (same batching as fmod test).
    Pairs where the torch (reference) result is NaN or Inf are SKIPPED from accuracy
    comparison.
    """
    ttnn_dtype = ttnn.float32
    N_base = base_values.numel()
    N_exp = exponent_values.numel()
    total_pairs = N_base * N_exp
    print(f"\n{'='*60}")
    print(f"{test_name} (float32): Testing {N_base:,} bases × {N_exp:,} exponents = {total_pairs:,} pairs ")

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

    exp_full = exponent_values.to(torch.float32).unsqueeze(0).clone()  # [1, N_exp]
    # Flush input exponents: subnormal and non-finite to zero (once, reused for all batches)
    flush_subnormal_values_to_zero(exp_full)
    exp_full[~torch.isfinite(exp_full)] = 0.0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N_base)
        x_batch_f32 = base_values[start:end].to(torch.float32).unsqueeze(1).clone()  # [B, 1]
        # Flush input bases: subnormal and non-finite to zero
        flush_subnormal_values_to_zero(x_batch_f32)
        x_batch_f32[~torch.isfinite(x_batch_f32)] = 0.0
        z_torch = torch.pow(x_batch_f32, exp_full)  # [B, N_exp]

        x_tt = ttnn.from_torch(x_batch_f32, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        exp_tt = ttnn.from_torch(exp_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.pow(x_tt, exp_tt)
        tt_out = ttnn.to_torch(z_tt)

        z_f32 = z_torch.to(torch.float32).contiguous()
        tt_f32 = tt_out.to(torch.float32).contiguous()
        valid_mask = torch.isfinite(z_f32)
        skipped_mask = ~valid_mask
        total_skipped_pairs += skipped_mask.sum().item()

        if skipped_mask.any() and len(skipped_samples) < max_skipped_samples:
            skipped_indices = skipped_mask.nonzero(as_tuple=False)
            for idx_pair in skipped_indices:
                if len(skipped_samples) >= max_skipped_samples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                reason = "NaN" if torch.isnan(z_f32[i, j]) else ("+Inf" if z_f32[i, j] > 0 else "-Inf")
                skipped_samples.append(
                    {
                        "base": x_batch_f32[i, 0].item(),
                        "exponent": exp_full[0, j].item(),
                        "torch_result": z_f32[i, j].item(),
                        "ttnn_result": tt_f32[i, j].item(),
                        "skip_reason": reason,
                        "category": variant_name,
                    }
                )

        # Flush subnormal and non-finite (outputs) to zero
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
        max_ulp_batch = ulp_dist[valid_mask].max().item() if valid_count > 0 else 0
        max_ulp_global = max(max_ulp_global, max_ulp_batch)
        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()
        mismatch_count = ((ulp_dist > 1) & valid_mask).sum().item()
        total_mismatches += mismatch_count
        total_valid_pairs += valid_count
        total_batch_pairs = x_batch_f32.shape[0] * exp_full.shape[1]

        if progress_every and ((batch_idx + 1) % progress_every == 0 or batch_idx == num_batches - 1):
            print(
                f"  Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, tested={valid_count}/{total_batch_pairs}"
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


# FP32 exhaustive tests (unary pow: all bases × all exponents, same normal finite non-zero set)
def test_pow_tt_FP32_exhaustive_pos_pos(device):
    """Exhaustive: positive base ^ positive exponent (all normal finite non-zero)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_float32_exhaustive_exponents(
        device,
        pos_values,
        pos_values,
        "Positive base ^ Positive exponent (exhaustive)",
        "pos_pos",
    )


def test_pow_tt_FP32_exhaustive_pos_neg(device):
    """Exhaustive: positive base ^ negative exponent (all normal finite non-zero)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_float32_exhaustive_exponents(
        device,
        pos_values,
        neg_values,
        "Positive base ^ Negative exponent (exhaustive)",
        "pos_neg",
    )


def test_pow_tt_FP32_exhaustive_neg_pos(device):
    """Exhaustive: negative base ^ positive exponent. Many NaN (non-integer exp); helper skips."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_float32_exhaustive_exponents(
        device,
        neg_values,
        pos_values,
        "Negative base ^ Positive exponent (exhaustive)",
        "neg_pos",
    )


def test_pow_tt_FP32_exhaustive_neg_neg(device):
    """Exhaustive: negative base ^ negative exponent. Many NaN; helper skips."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_float32_exhaustive_exponents(
        device,
        neg_values,
        neg_values,
        "Negative base ^ Negative exponent (exhaustive)",
        "neg_neg",
    )


@pytest.mark.timeout(1800)
def test_pow_tt_FP32_exhaustive_test(device):
    """Unary pow (base**exponent) with fp32 - all sign combinations, one exponent per variant."""
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


# BF16 Exhaustive tests (unary pow: all bases × all exponents, same normal finite non-zero set)
def test_pow_tt_bf16_exhaustive_pos_pos(device):
    """Exhaustive: positive base ^ positive exponent (all normal finite non-zero)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_exhaustive_exponents(
        device,
        pos_values,
        pos_values,
        "Positive base ^ Positive exponent (exhaustive)",
        "pos_pos",
    )


def test_pow_tt_bf16_exhaustive_pos_neg(device):
    """Exhaustive: positive base ^ negative exponent (all normal finite non-zero)."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_exhaustive_exponents(
        device,
        pos_values,
        neg_values,
        "Positive base ^ Negative exponent (exhaustive)",
        "pos_neg",
    )


def test_pow_tt_bf16_exhaustive_neg_pos(device):
    """Exhaustive: negative base ^ positive exponent. Many NaN; helper skips."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_exhaustive_exponents(
        device,
        neg_values,
        pos_values,
        "Negative base ^ Positive exponent (exhaustive)",
        "neg_pos",
    )


def test_pow_tt_bf16_exhaustive_neg_neg(device):
    """Exhaustive: negative base ^ negative exponent. Many NaN; helper skips."""
    torch.manual_seed(0)
    pos_values, neg_values = _generate_normal_finite_nonzero_bf16_values()
    return _run_exhaustive_unary_pow_helper_exhaustive_exponents(
        device,
        neg_values,
        neg_values,
        "Negative base ^ Negative exponent (exhaustive)",
        "neg_neg",
    )


@pytest.mark.timeout(1800)
def test_pow_tt_BF16_exhaustive_test(device):
    """Unary pow (base**exponent) with BF16 - all sign combinations, one exponent per variant."""
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
