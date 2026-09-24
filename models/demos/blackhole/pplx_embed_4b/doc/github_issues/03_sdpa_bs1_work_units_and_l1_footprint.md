---
title: "SDPA (non-causal, GQA 32/8, S=512, d=128) on P150: batch-1 fills only 64 of 120 cores; batched static CBs leave no L1 for resident activations — Qwen3-4B prefill"
assignee: cmaryanTT
labels: perf, blackhole, sdpa
---

## Part 1 — batch 1: 55 µs = 78 TFLOP/s, 64 work units

`ttnn.transformer.scaled_dot_product_attention(is_causal=False)` on Q `[1,32,512,128]`,
K/V `[1,8,512,128]` (bfp8_b in L1), LoFi, `fp32_dest_acc_en=False` (streaming kernel),
`exp_approx_mode=True`, traced on one P150 (12×10 grid):

| grid | q_chunk / k_chunk | µs |
|---|---|---|
| 8×8 | 512 / 256 | 79.1 |
| **8×8** | **256 / 256** | **55.0** |
| 8×10 | 256 / 256 | 55.4 |
| 8×8 | 256 / 128 | 58.9 |
| 12×10 | 256 / 256 | 67.6 |
| 8×8 | 128 / 128 | 72.2 |
| 8×8 | 512 / 256, fp32 acc on (legacy kernel) | 109.3 |

Work units are `batch × heads × ceil(S / q_chunk)` = 64 at q256, so 56 of 120 cores idle, and a
larger grid only adds setup cost. 4.3 GFLOP in 55 µs is 78 TFLOP/s; the same chip runs the
adjacent matmuls at 320–470 TFLOP/s. For Qwen3-4B at batch 1 this op is 36 × 55 µs ≈ 2.0 ms of a
17.7 ms forward.

**Ask:** a way to fill 120 cores at this size without the q128 penalty — e.g. splitting a head's
K/V range across two cores with an online-softmax merge, or making the per-chunk fixed cost small
enough that q128 (128 units) wins. Target: ≤ 35 µs for this shape.

## Part 2 — batched: SDPA / head-split static CBs vs L1-resident activations

At M = 4096 (batch 8 × 512) and M = 8192, keeping the Q/K/V heads (27 MB → 223 KB/core at batch 8)
or the QKV projection output in L1 fails trace capture:

```
TT_THROW: Statically allocated circular buffers in program 287 clash with L1 buffers on core range
[0-0 - 11-9]. L1 buffer allocated at 1462656 and static circular buffer region ends at 1505408
```
(42 KB/core short with k_chunk 512; 28 KB short with k_chunk 256: `1073152` vs `1101952`). The
same placements are worth 10–20% on BGE-M3 at B8/B16 (dim 1024), and the only configuration that
fits here (heads only + k256) gives −0.7% at batch 8.

**Ask:** a smaller static-CB footprint for the 12×10 / k512 configuration (single-buffered K/V,
smaller out CB, or a program-config knob bounding CB bytes), so ≈ 250 KB/core of L1 stays free
for activations while SDPA runs.

## Repro (Part 1)

```python
import time, torch, ttnn
B8 = ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
def CK(f32): return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                                     fp32_dest_acc_en=f32, packer_l1_acc=True)
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
    L1 = ttnn.L1_MEMORY_CONFIG
    q = ttnn.from_torch(torch.randn(1, 32, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    k = ttnn.from_torch(torch.randn(1, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    v = ttnn.from_torch(torch.randn(1, 8, 512, 128), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    for gx, gy in ((8, 8), (8, 10), (12, 10)):
        for qc in (512, 256, 128):
            for kc in (512, 256, 128):
                pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                                            q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=True)
                try:
                    us = traced(lambda: ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, is_causal=False, scale=0.088388346, program_config=pc,
                        compute_kernel_config=CK(False), memory_config=L1))
                    print(f"grid {gx}x{gy} q{qc}/k{kc}: {us:7.1f} us  ({4.29e9/us/1e6:4.0f} TFLOP/s)")
                except Exception as e:
                    print(f"grid {gx}x{gy} q{qc}/k{kc}: failed {str(e).splitlines()[0][:80]}")
finally:
    ttnn.close_device(D)
```
