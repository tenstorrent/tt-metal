### Issue
In Qwen3-4B batched prefill on a P150 the two residual RMSNorms per layer run `ttnn.rms_norm` on `[1, 1, 4096, 2560]` bfp8_b in DRAM → bfp8_b in DRAM, 120 cores, HiFi2 with `fp32_dest_acc_en=True`, and take **90 µs per call** (device kernel duration). The tensor is 10.6 MB; the kernel reads it twice (sum of squares, then normalise) and writes it once ≈ 32 MB ≈ 350 GB/s, while a single-pass kernel would move 21 MB ≈ 47 µs at that bandwidth. This is 36 × 2 × 90 µs ≈ 6.5 ms of a 123 ms forward at batch 8 and scales with M (≈ 13 ms at batch 16, ≈ 26 ms at batch 32).

Measured end to end and not helpful: output in L1 (+1.4% at batch 8), `fp32_dest_acc_en=False` (−0.7% at batch 8, +1.2% at batch 32), LoFi + approx (kernel −1.5%). Block-sharded LN does not fit L1 at these shapes next to the matmul CBs.

### Expected
A single-pass interleaved RMSNorm for wide rows (2560 = 80 tiles) that keeps a row block in L1 across the reduce and the normalise, or a fused residual-add + RMSNorm that emits both the sum and the normalised output (the decoder needs both). Target: ≤ 55 µs at M = 4096, PCC ≥ 0.9999 vs the current kernel.

### Unit test (random data, one P150)
Passes when M = 4096 is ≤ 55 µs.

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
    ok = True
    for M in (4096, 8192, 16384):
        x = ttnn.from_torch(torch.randn(1, 1, M, 2560), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.from_torch(torch.rand(1, 1, 1, 2560) + 0.5, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=D)
        us = traced(lambda: ttnn.rms_norm(x, epsilon=1e-6, weight=g, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        print(f"M={M:5d}: {us:7.1f} us   {2*M*2560*1.0625/us/1e3:5.0f} GB/s single-pass-equivalent (DRAM peak ~450)")
        if M == 4096: ok = us <= 55
        ttnn.deallocate(x); ttnn.deallocate(g)
    print("PASS" if ok else "FAIL")
finally:
    ttnn.close_device(D)
```
