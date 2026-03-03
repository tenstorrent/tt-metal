# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import os
import csv
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp

# BF16 exhaustive: pytest /home/ubuntu/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_xielu_accuracy.py::test_xielu_tt_BF16_exhaustive_test
# FP32 exhaustive: pytest /home/ubuntu/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_xielu_accuracy.py::test_xielu_tt_FP32_exhaustive_test

pytestmark = pytest.mark.use_module_device

ALPHA_P = 0.8
ALPHA_N = 0.8


def flush_subnormal_values(tensor):
    SUBNORMAL_THRESHOLD = 2.0 ** (-126)
    mask = torch.abs(tensor) < SUBNORMAL_THRESHOLD
    tensor[mask] = 0.0
    return tensor


def _run_exhaustive_xielu_helper(device, input_values, test_name, variant_name, max_skipped_samples=10):
    """Run exhaustive xielu test (BF16). Single input tensor; alpha_p=0.8, alpha_n=0.8.
    Skips pairs where torch reference is NaN/Inf. Returns skipped samples."""
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    ttnn_dtype = ttnn.bfloat16

    N = input_values.numel()
    print(f"\n{'='*60}")
    print(f"{test_name} (bfloat16): Testing {N:,} inputs (alpha_p={ALPHA_P}, alpha_n={ALPHA_N})")

    batch_size = 512
    num_batches = (N + batch_size - 1) // batch_size

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

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        x_batch = input_values[start:end].clone()
        flush_subnormal_values(x_batch)
        x_batch[~torch.isfinite(x_batch)] = 0.0

        z_torch = golden_fn(x_batch, alpha_p=ALPHA_P, alpha_n=ALPHA_N)
        x_2d = x_batch.unsqueeze(0) if x_batch.dim() == 1 else x_batch
        x_tt = ttnn.from_torch(x_2d, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.xielu(x_tt, alpha_p=ALPHA_P, alpha_n=ALPHA_N)
        tt_out = ttnn.to_torch(z_tt)
        if tt_out.dim() > 1:
            tt_out = tt_out.squeeze(0)

        z_bf16 = z_torch.to(torch.bfloat16).contiguous().flatten()
        tt_bf16 = tt_out.to(torch.bfloat16).contiguous().flatten()

        valid_mask = torch.isfinite(z_bf16)
        skipped_mask = ~valid_mask
        batch_skipped = skipped_mask.sum().item()
        total_skipped_pairs += batch_skipped

        if batch_skipped > 0 and len(skipped_samples) < max_skipped_samples:
            for i in range(z_bf16.numel()):
                if len(skipped_samples) >= max_skipped_samples:
                    break
                if not valid_mask[i]:
                    reason = "NaN" if torch.isnan(z_bf16[i]) else ("+Inf" if z_bf16[i] > 0 else "-Inf")
                    skipped_samples.append(
                        {
                            "input": x_batch.flatten()[i].item(),
                            "torch_result": z_bf16[i].item(),
                            "ttnn_result": tt_bf16[i].item(),
                            "skip_reason": reason,
                            "category": variant_name,
                        }
                    )

        flush_subnormal_values(z_bf16)
        flush_subnormal_values(tt_bf16)
        tt_bf16[~torch.isfinite(tt_bf16)] = 0.0

        z_bits = z_bf16.view(torch.uint16).to(torch.int32)
        tt_bits = tt_bf16.view(torch.uint16).to(torch.int32)
        sign_z = (z_bits >> 15) & 1
        sign_tt = (tt_bits >> 15) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x8000, 0x8000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x8000, 0x8000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()
        ulp_dist_valid = torch.where(valid_mask, ulp_dist, torch.zeros_like(ulp_dist))

        max_ulp_batch = ulp_dist_valid.max().item()
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()
        mismatch_count = ((ulp_dist > 1) & valid_mask).sum().item()
        total_mismatches += mismatch_count
        total_valid_pairs += valid_mask.sum().item()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, skipped={batch_skipped}"
            )

    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    skipped_pct = (total_skipped_pairs / N) * 100 if N > 0 else 0.0

    print(f"\n--- Summary ---")
    print(f"Total inputs: {N:,}")
    print(f"Valid (torch finite): {total_valid_pairs:,} ({100 - skipped_pct:.4f}%)")
    print(f"Skipped (torch NaN/Inf): {total_skipped_pairs:,} ({skipped_pct:.4f}%)")
    print(f"Skipped samples collected: {len(skipped_samples)}")
    print(f"Global max ULP: {max_ulp_global}")
    print(f"Total mismatches (ULP > 1): {total_mismatches:,} ({mismatch_pct:.4f}% of valid)")
    print(f"\nULP Distribution (valid only):")
    if total_valid_pairs > 0:
        print(f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_pairs*100:.4f}%)")
    return skipped_samples


def _run_exhaustive_xielu_helper_float32(device, input_values, test_name, variant_name, max_skipped_samples=10):
    """Run exhaustive xielu test (float32). Single input tensor; alpha_p=0.8, alpha_n=0.8."""
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    ttnn_dtype = ttnn.float32

    N = input_values.numel()
    print(f"\n{'='*60}")
    print(f"{test_name} (float32): Testing {N:,} inputs (alpha_p={ALPHA_P}, alpha_n={ALPHA_N})")

    batch_size = 512
    num_batches = (N + batch_size - 1) // batch_size

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

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        x_batch = input_values[start:end].to(torch.float32).clone()
        flush_subnormal_values(x_batch)
        x_batch[~torch.isfinite(x_batch)] = 0.0

        z_torch = golden_fn(x_batch, alpha_p=ALPHA_P, alpha_n=ALPHA_N)
        x_2d = x_batch.unsqueeze(0) if x_batch.dim() == 1 else x_batch
        x_tt = ttnn.from_torch(x_2d, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.xielu(x_tt, alpha_p=ALPHA_P, alpha_n=ALPHA_N)
        tt_out = ttnn.to_torch(z_tt)
        if tt_out.dim() > 1:
            tt_out = tt_out.squeeze(0)

        z_f32 = z_torch.to(torch.float32).contiguous().flatten()
        tt_f32 = tt_out.to(torch.float32).contiguous().flatten()

        valid_mask = torch.isfinite(z_f32)
        skipped_mask = ~valid_mask
        batch_skipped = skipped_mask.sum().item()
        total_skipped_pairs += batch_skipped

        if batch_skipped > 0 and len(skipped_samples) < max_skipped_samples:
            for i in range(z_f32.numel()):
                if len(skipped_samples) >= max_skipped_samples:
                    break
                if not valid_mask[i]:
                    reason = "NaN" if torch.isnan(z_f32[i]) else ("+Inf" if z_f32[i] > 0 else "-Inf")
                    skipped_samples.append(
                        {
                            "input": x_batch.flatten()[i].item(),
                            "torch_result": z_f32[i].item(),
                            "ttnn_result": tt_f32[i].item(),
                            "skip_reason": reason,
                            "category": variant_name,
                        }
                    )

        flush_subnormal_values(z_f32)
        flush_subnormal_values(tt_f32)
        tt_f32[~torch.isfinite(tt_f32)] = 0.0

        z_bits = z_f32.view(torch.int32)
        tt_bits = tt_f32.view(torch.int32)
        sign_z = (z_bits >> 31) & 1
        sign_tt = (tt_bits >> 31) & 1
        z_ord = torch.where(sign_z == 0, z_bits + 0x80000000, 0x80000000 - z_bits)
        tt_ord = torch.where(sign_tt == 0, tt_bits + 0x80000000, 0x80000000 - tt_bits)
        ulp_dist = (z_ord - tt_ord).abs()
        ulp_dist_valid = torch.where(valid_mask, ulp_dist, torch.zeros_like(ulp_dist))

        max_ulp_batch = ulp_dist_valid.max().item()
        max_ulp_global = max(max_ulp_global, max_ulp_batch)

        ulp_0_count += ((ulp_dist == 0) & valid_mask).sum().item()
        ulp_1_count += ((ulp_dist == 1) & valid_mask).sum().item()
        ulp_2_count += ((ulp_dist == 2) & valid_mask).sum().item()
        ulp_3_to_10_count += ((ulp_dist >= 3) & (ulp_dist <= 10) & valid_mask).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100) & valid_mask).sum().item()
        ulp_above_100_count += ((ulp_dist > 100) & valid_mask).sum().item()
        mismatch_count = ((ulp_dist > 1) & valid_mask).sum().item()
        total_mismatches += mismatch_count
        total_valid_pairs += valid_mask.sum().item()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: max_ulp={max_ulp_batch}, mismatches={mismatch_count}, skipped={batch_skipped}"
            )

    mismatch_pct = (total_mismatches / total_valid_pairs) * 100 if total_valid_pairs > 0 else 0.0
    skipped_pct = (total_skipped_pairs / N) * 100 if N > 0 else 0.0

    print(f"\n--- Summary ---")
    print(f"Total inputs: {N:,}")
    print(f"Valid (torch finite): {total_valid_pairs:,} ({100 - skipped_pct:.4f}%)")
    print(f"Skipped (torch NaN/Inf): {total_skipped_pairs:,} ({skipped_pct:.4f}%)")
    print(f"Skipped samples collected: {len(skipped_samples)}")
    print(f"Global max ULP: {max_ulp_global}")
    print(f"Total mismatches (ULP > 1): {total_mismatches:,} ({mismatch_pct:.4f}% of valid)")
    print(f"\nULP Distribution (valid only):")
    if total_valid_pairs > 0:
        print(f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_valid_pairs*100:.4f}%)")
        print(f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/total_valid_pairs*100:.4f}%)")
    return skipped_samples


def _get_pos_neg_bf16_values():
    """All normal finite non-zero bf16 values, split into positive and negative."""
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals = all_bitpatterns.view(torch.bfloat16)
    tiny = torch.finfo(torch.bfloat16).tiny
    pos_mask = torch.isfinite(vals) & (vals > 0) & (vals >= tiny)
    neg_mask = torch.isfinite(vals) & (vals < 0) & (vals.abs() >= tiny)
    return vals[pos_mask], vals[neg_mask]


# --- FP32 exhaustive ---
def test_xielu_tt_FP32_exhaustive_pos_input(device):
    """xielu FP32: positive inputs only, alpha_p=0.8, alpha_n=0.8."""
    torch.manual_seed(0)
    pos_values, neg_values = _get_pos_neg_bf16_values()
    return _run_exhaustive_xielu_helper_float32(
        device,
        pos_values.to(torch.float32),
        "xielu positive input",
        "pos_input",
    )


def test_xielu_tt_FP32_exhaustive_neg_input(device):
    """xielu FP32: negative inputs only, alpha_p=0.8, alpha_n=0.8."""
    torch.manual_seed(0)
    pos_values, neg_values = _get_pos_neg_bf16_values()
    return _run_exhaustive_xielu_helper_float32(
        device,
        neg_values.to(torch.float32),
        "xielu negative input",
        "neg_input",
    )


@pytest.mark.timeout(1800)
def test_xielu_tt_FP32_exhaustive_test(device):
    """xielu FP32 exhaustive: positive and negative; writes results to ~/Files/."""
    import io
    import sys
    from datetime import datetime

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

    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, captured_output)
    all_skipped = []
    try:
        all_skipped.extend(test_xielu_tt_FP32_exhaustive_pos_input(device))
        all_skipped.extend(test_xielu_tt_FP32_exhaustive_neg_input(device))
    finally:
        sys.stdout = original_stdout

    out_dir = os.path.join(os.path.expanduser("~"), "Files")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"xielu_fp32_exhaustive_results_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(captured_output.getvalue())

    skipped_csv_path = os.path.join(out_dir, f"xielu_fp32_skipped_samples_{timestamp}.csv")
    with open(skipped_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "input", "torch_result", "ttnn_result", "skip_reason"])
        w.writeheader()
        w.writerows(all_skipped)

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")
    print(f"Skipped samples CSV: {skipped_csv_path}")
    print(f"{'='*60}")


# --- BF16 exhaustive ---
def test_xielu_tt_BF16_exhaustive_pos_input(device):
    """xielu BF16: positive inputs only, alpha_p=0.8, alpha_n=0.8."""
    torch.manual_seed(0)
    pos_values, neg_values = _get_pos_neg_bf16_values()
    return _run_exhaustive_xielu_helper(
        device,
        pos_values,
        "xielu positive input",
        "pos_input",
    )


def test_xielu_tt_BF16_exhaustive_neg_input(device):
    """xielu BF16: negative inputs only, alpha_p=0.8, alpha_n=0.8."""
    torch.manual_seed(0)
    pos_values, neg_values = _get_pos_neg_bf16_values()
    return _run_exhaustive_xielu_helper(
        device,
        neg_values,
        "xielu negative input",
        "neg_input",
    )


@pytest.mark.timeout(1800)
def test_xielu_tt_BF16_exhaustive_test(device):
    """xielu BF16 exhaustive: positive and negative; writes results to ~/Files/."""
    import io
    import sys
    from datetime import datetime

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

    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, captured_output)
    all_skipped = []
    try:
        all_skipped.extend(test_xielu_tt_BF16_exhaustive_pos_input(device))
        all_skipped.extend(test_xielu_tt_BF16_exhaustive_neg_input(device))
    finally:
        sys.stdout = original_stdout

    out_dir = os.path.join(os.path.expanduser("~"), "Files")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"xielu_bf16_exhaustive_results_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(captured_output.getvalue())

    skipped_csv_path = os.path.join(out_dir, f"xielu_bf16_skipped_samples_{timestamp}.csv")
    with open(skipped_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "input", "torch_result", "ttnn_result", "skip_reason"])
        w.writeheader()
        w.writerows(all_skipped)

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")
    print(f"Skipped samples CSV: {skipped_csv_path}")
    print(f"{'='*60}")
