### Issue
`ttnn.transformer.scaled_dot_product_attention(is_causal=False)` for Qwen3-4B GQA (Q `[1,32,512,128]`, K/V `[1,8,512,128]`, bfp8_b in L1, LoFi, `fp32_dest_acc_en=False` = streaming kernel, `exp_approx_mode=True`) on a P150 12×10 grid:

| grid | q_chunk / k_chunk | µs |
|---|---|---|
| 8×8 | 512 / 256 | 79.1 |
| **8×8** | **256 / 256** | **55.0** |
| 8×10 | 256 / 256 | 55.4 |
| 8×8 | 256 / 128 | 58.9 |
| 12×10 | 256 / 256 | 67.6 |
| 8×8 | 128 / 128 | 72.2 |
| 8×8 | 512 / 256, fp32 acc on (legacy kernel) | 109.3 |

Work units are `batch × heads × ceil(S / q_chunk)` = 64 at q256, so 56 of 120 cores idle and a larger grid only adds setup cost; 4.3 GFLOP in 55 µs is 78 TFLOP/s next to matmuls at 320–470. At batch 1 this is 36 × 55 µs ≈ 2.0 ms of a 17.7 ms forward.

Second, batched: at M = 4096 / 8192 (batch 8 / 16 × 512) the SDPA (12×10, k512) and head-split static CBs leave no L1 for resident activations. Keeping the Q/K/V heads (223 KB/core at batch 8) or the QKV output in L1 fails trace capture:
```
TT_THROW: Statically allocated circular buffers in program 287 clash with L1 buffers on core range [0-0 - 11-9].
L1 buffer allocated at 1462656 and static circular buffer region ends at 1505408
```
(42 KB/core short at k512, 28 KB short at k256). The same placements are worth 10–20% on BGE-M3 at B8/B16.

### Expected
1. Batch 1: ≤ 35 µs for this shape by filling 120 cores — e.g. split a head's K/V range over two cores with an online-softmax merge, or make the per-chunk fixed cost small enough that q128 (128 units) wins.
2. Batched: a smaller static-CB footprint for 12×10 / k512 (single-buffered K/V, smaller out CB, or a config knob bounding CB bytes) so ≈ 250 KB/core stays free while SDPA runs.

### Unit test (random data, one P150)
Part 1 sweep; passes when the best configuration is ≤ 35 µs.

```python
import time, torch, ttnn
B8 = ttnn.bfloat8_b
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)
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
    best = float("inf")
    for gx, gy in ((8, 8), (8, 10), (12, 10)):
        for qc in (512, 256, 128):
            for kc in (512, 256, 128):
                pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                                            q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=True)
                try:
                    us = traced(lambda: ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, is_causal=False, scale=0.088388346, program_config=pc,
                        compute_kernel_config=ckc, memory_config=L1))
                    best = min(best, us)
                    print(f"grid {gx}x{gy} q{qc}/k{kc}: {us:7.1f} us  ({4.29e9/us/1e6:4.0f} TFLOP/s)")
                except Exception as e:
                    print(f"grid {gx}x{gy} q{qc}/k{kc}: failed {str(e).splitlines()[0][:80]}")
    print("PASS" if best <= 35 else f"FAIL: best {best:.1f} us")
finally:
    ttnn.close_device(D)
```
