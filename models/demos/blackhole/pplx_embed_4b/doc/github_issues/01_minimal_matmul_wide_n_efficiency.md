---
title: "minimal_matmul: (K=2560, N=9728) runs at 332 TFLOP/s while (K=9728, N=2560) runs at 473 on P150 — Qwen3-4B FF1/FF3 prefill"
assignee: sankarmanoj-tt
labels: perf, blackhole, matmul
---

## Summary

For Qwen3-4B prefill on a Blackhole P150 (12×10 worker grid), `ttnn.experimental.minimal_matmul`
reaches very different efficiency on two matmuls with identical FLOPs per row:

| shape (M × K × N) | config (12×10) | µs | TFLOP/s |
|---|---|---|---|
| 4096 × 2560 × 9728 (FF1 / FF3) | blk 8,8,8 sb 1×8 | 615 | **332** |
| 4096 × 9728 × 2560 (FF2) | blk 16,8,8 sb 1×8 | 431 | **473** |
| 8192 × 2560 × 6144 (QKV) | blk 8,8,8 sb 1×8 | 595 | 433 |
| 8192 × 4096 × 2560 (WO) | blk 8,8,8 sb 1×8 | 381 | 451 |

Weights bfp4_b, DRAM width-sharded over the 8 banks (also measured with interleaved: same
picture); activations bfp8_b in DRAM; LoFi, `fp32_dest_acc_en=False`, `packer_l1_acc=True`.
A block sweep (M 4–16, K 4–20, N 4–8, subblocks 1×8 / 2×4 / 1×4) moves FF1 by ≤ 6%. The legacy
2D-multicast kernel is 3–60% slower on these shapes, so `minimal_matmul` is the right kernel; the
question is why the wide-N / short-K shape loses 30% per FLOP.

Hypothesis: with K = 80 tiles the K loop is short and in0 (M × K, 10.6 MB at M=4096) is
re-streamed from DRAM once per N block (38 N blocks of 8 tiles at N=304 tiles) — ≈ 400 MB of in0
traffic per call, i.e. ≈ 650 GB/s at 615 µs, above the DRAM roofline — while FF2 (K = 304 tiles,
N = 80) re-streams its in0 only 10 times.

## Why it matters

At M = 16384 (batch 32 × 512 tokens) FF1 + FF3 are ≈ 177 ms of a 438 ms forward. Reaching FF2's
473 TFLOP/s on them is ≈ −53 ms (−12%); at M = 4096 it is ≈ −6 ms of 123.

## Ask

1. Confirm the bound (in0 re-read vs. output write vs. blocking) for the wide-N shape.
2. A dataflow or blocking change (e.g. keep an M × K_block in0 slab resident across N blocks,
   or multicast in0 along the grid row) that brings (K=2560, N=9728) to ≥ 430 TFLOP/s.

## Repro / acceptance test

Runs on one P150 (`TT_VISIBLE_DEVICES=<id>`), prints µs and TFLOP/s per shape, checks PCC vs torch.
Acceptance: FF1 shape ≥ 430 TFLOP/s at M=4096 and M=16384 with PCC ≥ 0.999.

```python
import math, time, torch, ttnn
B4, B8, T = ttnn.bfloat4_b, ttnn.bfloat8_b, 32
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)
banks = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])

def sharded_w(wt, K, N):
    pad = math.ceil(N / (T * 8)) * (T * 8)
    spec = ttnn.ShardSpec(banks, (K, pad // 8), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    return ttnn.from_torch(wt, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc)

def traced(fn, n=2, reps=4):
    for _ in range(2): ttnn.deallocate(fn())
    ttnn.synchronize_device(D); tid = ttnn.begin_trace_capture(D, cq_id=0)
    outs = [fn() for _ in range(n)]; ttnn.end_trace_capture(D, tid, cq_id=0)
    ttnn.execute_trace(D, tid, cq_id=0, blocking=True); t0 = time.perf_counter()
    for _ in range(reps): ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / reps / n * 1e6; ttnn.release_trace(D, tid)
    for o in outs: ttnn.deallocate(o)
    return us

def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

try:
    for M in (4096, 16384):
        for name, K, N, blk in (("FF1", 2560, 9728, (8, 8, 8, 1, 8)), ("FF2", 9728, 2560, (16, 8, 8, 1, 8))):
            xt = torch.randn(1, 1, M, K); wt = torch.randn(1, 1, K, N) * 0.02
            x = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w = sharded_w(wt, K, N)
            mb, kb, nb, sh, sw = blk
            cfg = ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh,
                                           subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(12, 10))
            run = lambda: ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg,
                                                           memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8)
            ref = ttnn.to_torch(x).float() @ ttnn.to_torch(w).float()
            p = pcc(ttnn.to_torch(run()), ref)
            us = traced(run)
            print(f"M={M:5d} {name} K={K} N={N}: {us:8.1f} us  {2*M*K*N/us/1e6:5.0f} TFLOP/s  pcc={p:.5f}")
            ttnn.deallocate(x); ttnn.deallocate(w)
finally:
    ttnn.close_device(D)
```
