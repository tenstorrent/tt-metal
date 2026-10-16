---
title: "rms_norm (interleaved, 4096×2560 bfp8) runs at ~50% of DRAM bandwidth on P150 — Qwen3-4B batched prefill"
assignee: cmaryanTT
labels: perf, blackhole, normalization
---

## Summary

In Qwen3-4B batched prefill on a Blackhole P150 the two residual RMSNorms per layer run on
`[1, 1, 4096, 2560]` bfp8_b in DRAM → bfp8_b in DRAM, 120 cores, HiFi2 with
`fp32_dest_acc_en=True`, and take **90 µs per call** (device kernel duration, traced replay).
The tensor is 10.6 MB; a two-pass kernel (sum of squares, then normalise) reads it twice and writes
it once ≈ 32 MB → ≈ 350 GB/s, and a single-pass kernel would move 21 MB → ≈ 47 µs at the same
bandwidth. The op is 36 × 2 × 90 µs ≈ 6.5 ms of a 123 ms forward at batch 8, and scales with M
(≈ 13 ms at batch 16, ≈ 26 ms at batch 32).

Things that did not help, measured same-chip end to end: output in L1 (+1.4% at batch 8),
`fp32_dest_acc_en=False` (−0.7% at batch 8, +1.2% at batch 32), LoFi + approx (kernel −1.5%).
Block-sharded LN does not fit L1 at these shapes (16+ tiles per core against the matmul CBs).

## Ask

A single-pass interleaved RMSNorm for wide rows (2560 = 80 tiles) that holds a row block in L1
across the reduce and the normalise, or a fused residual-add + RMSNorm variant that emits both the
sum and the normalised output (both are needed by the decoder). Target: ≤ 55 µs for this shape.

## Repro / acceptance

```python
import time, torch, ttnn
B8 = ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
def traced(fn, n=4, reps=8):
    for _ in range(2): ttnn.deallocate(fn())
    ttnn.synchronize_device(D); tid = ttnn.begin_trace_capture(D, cq_id=0)
    outs = [fn() for _ in range(n)]; ttnn.end_trace_capture(D, tid, cq_id=0)
    ttnn.execute_trace(D, tid, cq_id=0, blocking=True); t0 = time.perf_counter()
    for _ in range(reps): ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / reps / n * 1e6; ttnn.release_trace(D, tid)
    for o in outs: ttnn.deallocate(o)
    return us
try:
    for M in (4096, 8192, 16384):
        xt = torch.randn(1, 1, M, 2560)
        x = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.from_torch(torch.rand(1, 1, 1, 2560) + 0.5, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=D)
        run = lambda: ttnn.rms_norm(x, epsilon=1e-6, weight=g, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        us = traced(run)
        bytes_1pass = 2 * M * 2560 * 1.0625
        print(f"M={M:5d}: {us:7.1f} us   {bytes_1pass/us/1e3:5.0f} GB/s single-pass-equivalent (DRAM peak ~450)")
        ttnn.deallocate(x); ttnn.deallocate(g)
finally:
    ttnn.close_device(D)
```
Acceptance: M=4096 ≤ 55 µs with PCC ≥ 0.9999 vs the current kernel.
