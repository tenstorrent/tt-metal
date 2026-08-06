"""The p150's two ceilings, measured, so "how much of the chip do we use" is a fact not a guess.

bw_vs_rows.py showed a batch-1 matmul is bandwidth/overhead bound (rows 1->32 cost the SAME time,
so 32x the FLOPs are free) and a 4096-row one is compute bound. Those are the two ceilings:

  * STREAMING BANDWIDTH -- a big elementwise binary op is the closest thing to a pure DRAM stream:
    read a + read b + write out, no reuse, no reduction. `clone` was NOT this (why_only_7pct.py
    got 138 GB/s from it, below what Block 1 demonstrably sustains, so clone is latency-bound).
  * PEAK COMPUTE -- a large square matmul, where DRAM traffic per FLOP goes to ~0.

Then: what fraction of each does a decode frame actually use?
"""
import time

import torch
import ttnn

from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def bench(dev, fn, reps=30):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps


def main():
    dev = open_device()
    try:
        print("=== STREAMING BANDWIDTH (elementwise: read a + read b + write out) ===")
        best_bw = 0.0
        for n in (2048, 4096, 8192):
            a = ttnn.from_torch(torch.randn(1, 1, n, n), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            b = ttnn.from_torch(torch.randn(1, 1, n, n), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s = bench(dev, lambda: ttnn.add(a, b))
            gbs = n * n * 2 * 3 / s / 1e9
            best_bw = max(best_bw, gbs)
            print(f"  add [{n}x{n}] bf16   {s*1e3:8.3f} ms   {gbs:6.0f} GB/s")
            del a, b

        print("\n=== PEAK COMPUTE (large square matmul) ===")
        best_tf = 0.0
        for n in (2048, 4096):
            a = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.02, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)
            w = ttnn.from_torch(torch.randn(n, n) * 0.02, dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=dev)
            s = bench(dev, lambda: ttnn.linear(a, w, compute_kernel_config=COMPUTE))
            tf = 2 * n * n * n / s / 1e12
            best_tf = max(best_tf, tf)
            print(f"  [{n}x{n}] @ [{n}x{n}]   {s*1e3:8.3f} ms   {tf:6.1f} TFLOP/s")
            del a, w

        print("\n=== what one decode frame actually uses ===")
        # from why_only_7pct.py, measured
        GB, MS = 6.698, 40.18
        params = GB * 1e9 / 1.0625                # bfloat8_b weights
        flops = 2 * params                        # matrix-vector: 2 FLOP per weight
        ach_bw = GB / (MS / 1e3) / 1e9
        ach_tf = flops / (MS / 1e3) / 1e12
        print(f"  bytes moved   {GB:.3f} GB in {MS:.2f} ms = {ach_bw:5.0f} GB/s "
              f"-> {ach_bw/best_bw*100:5.1f}% of the {best_bw:.0f} GB/s ceiling")
        print(f"  arithmetic    {flops/1e9:.1f} GFLOP in {MS:.2f} ms = {ach_tf:5.2f} TFLOP/s "
              f"-> {ach_tf/best_tf*100:5.2f}% of the {best_tf:.0f} TFLOP/s ceiling")
        print(f"\n  N150 same graph at ~48 ms = {GB/(48/1e3)/1e9:.0f} GB/s achieved.")
        print(f"  N150 ceiling ~200 GB/s -> it ran at ~{GB/(48/1e3)/1e9/200*100:.0f}% of ITS DRAM.")
        print(f"  p150 runs at {ach_bw/best_bw*100:.0f}% of its DRAM. Same graph, same bytes.")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
