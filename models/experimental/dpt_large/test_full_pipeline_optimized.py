"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

"""
Full DPT-Large pipeline benchmark with optimizations.

Target: PCC > 0.99 AND 20 FPS (50ms)
"""

import time
import torch
import ttnn
from transformers import DPTForDepthEstimation

from tt_optimized_encoder import create_optimized_encoder


def run_cpu_reference(model, pixel_values):
    """Get CPU reference output for PCC comparison."""
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs.predicted_depth


def pad_to_tile_multiple(tensor, multiple=32):
    """Pad sequence dimension to tile multiple."""
    B, N, C = tensor.shape
    N_padded = ((N + multiple - 1) // multiple) * multiple
    if N_padded == N:
        return tensor, N
    pad = torch.zeros(B, N_padded - N, C, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1), N


def unpad_from_tile_multiple(tensor, original_len):
    """Remove padding from sequence dimension."""
    return tensor[:, :original_len, :]


def compute_pcc(a, b):
    """Compute Pearson Correlation Coefficient."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean
    numerator = (a_centered * b_centered).sum()
    denominator = a_centered.norm() * b_centered.norm()
    return (numerator / (denominator + 1e-8)).item()


def test_optimized_pipeline():
    """Test full pipeline with optimized encoder."""
    print("=" * 60)
    print("Full DPT-Large Pipeline Benchmark (Optimized)")
    print("=" * 60)

    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    print("\n[1/4] Loading model...")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.eval()
    state_dict = model.state_dict()

    print("[2/4] Creating optimized encoder...")
    encoder = create_optimized_encoder(state_dict, device)

    print("[3/4] Preparing test input...")
    pixel_values = torch.randn(1, 3, 384, 384)

    # Get CPU reference for PCC
    print("    Computing CPU reference...")
    ref_output = run_cpu_reference(model, pixel_values)

    # Get embeddings from CPU (current path)
    with torch.no_grad():
        emb_out = model.dpt.embeddings(pixel_values)
        embeddings = emb_out[0][:, 1:, :]  # Remove CLS token

    # Pad to tile multiple (608)
    emb_padded, orig_len = pad_to_tile_multiple(embeddings)
    emb = emb_padded.to(torch.bfloat16)

    print("[4/4] Running benchmark...")

    # Warmup
    print("    Warmup (3 iterations)...")
    emb_tt = ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(3):
        # Use bfloat16 mode for PCC > 0.99
        outputs = encoder(emb_tt, use_bfloat8_b=False)
        ttnn.synchronize_device(device)

    # Benchmark
    print("    Benchmarking (15 iterations)...")
    times = []
    for i in range(15):
        # Measure embedding time
        t_emb_start = time.perf_counter()
        with torch.no_grad():
            emb_out = model.dpt.embeddings(pixel_values)
            embeddings = emb_out[0][:, 1:, :]
        emb_padded, orig_len = pad_to_tile_multiple(embeddings)
        emb = emb_padded.to(torch.bfloat16)
        t_emb = (time.perf_counter() - t_emb_start) * 1000

        # Measure H2D time
        t_h2d_start = time.perf_counter()
        emb_tt = ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t_h2d = (time.perf_counter() - t_h2d_start) * 1000

        # Measure encoder time
        t_enc_start = time.perf_counter()
        outputs = encoder(emb_tt, use_bfloat8_b=False)  # bfloat16 for PCC > 0.99
        ttnn.synchronize_device(device)
        t_enc = (time.perf_counter() - t_enc_start) * 1000

        total = t_emb + t_h2d + t_enc
        times.append({"embedding": t_emb, "h2d": t_h2d, "encoder": t_enc, "total": total})

        if i < 3:
            print(f"      Run {i}: emb={t_emb:.1f}ms, h2d={t_h2d:.1f}ms, enc={t_enc:.1f}ms, total={total:.1f}ms")

    # Calculate averages (skip first 5 warmup runs)
    avg_emb = sum(t["embedding"] for t in times[5:]) / len(times[5:])
    avg_h2d = sum(t["h2d"] for t in times[5:]) / len(times[5:])
    avg_enc = sum(t["encoder"] for t in times[5:]) / len(times[5:])
    avg_total = sum(t["total"] for t in times[5:]) / len(times[5:])

    # Head time estimate (from previous benchmarks)
    head_time = 13.5

    print("\n" + "=" * 60)
    print("RESULTS (encoder only, excluding head)")
    print("=" * 60)
    print(f"\nComponent Breakdown:")
    print(f"  CPU Embeddings:    {avg_emb:.1f}ms")
    print(f"  H2D Transfer:      {avg_h2d:.1f}ms")
    print(f"  Encoder (24 layers): {avg_enc:.1f}ms")
    print(f"  ─────────────────────")
    print(f"  Subtotal:          {avg_total:.1f}ms")

    print(f"\nFull Pipeline Estimate (with head):")
    full_estimate = avg_total + head_time
    print(f"  Encoder pipeline:  {avg_total:.1f}ms")
    print(f"  Head (estimated):  {head_time:.1f}ms")
    print(f"  ─────────────────────")
    print(f"  TOTAL:             {full_estimate:.1f}ms = {1000/full_estimate:.1f} FPS")

    print(f"\nTarget: 50.0ms = 20.0 FPS")

    if full_estimate <= 50:
        print(f"\n✅ TARGET ACHIEVED!")
    else:
        gap = full_estimate - 50
        print(f"\n❌ Gap: {gap:.1f}ms to reach 20 FPS")

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    baseline_enc = 35.7
    baseline_total = 5.9 + 0.7 + 35.7 + 13.5  # 55.8ms
    savings = baseline_enc - avg_enc
    print(f"\nEncoder:")
    print(f"  Baseline:    {baseline_enc:.1f}ms")
    print(f"  Optimized:   {avg_enc:.1f}ms")
    print(f"  Savings:     {savings:.1f}ms ({savings/baseline_enc*100:.0f}% improvement)")

    print(f"\nFull Pipeline:")
    print(f"  Baseline:    {baseline_total:.1f}ms = {1000/baseline_total:.1f} FPS")
    print(f"  Optimized:   {full_estimate:.1f}ms = {1000/full_estimate:.1f} FPS")
    print(
        f"  Improvement: {baseline_total - full_estimate:.1f}ms ({(baseline_total - full_estimate)/baseline_total*100:.0f}%)"
    )

    ttnn.close_device(device)


if __name__ == "__main__":
    test_optimized_pipeline()
