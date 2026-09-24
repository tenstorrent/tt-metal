---
title: "minimal_matmul fuse_swiglu: the epilogue halves the accumulation subblock and costs ~3× the SiLU itself; loses to unfused FF1+FF3+mul at M=16384 — Qwen3-4B prefill"
assignee: sankarmanoj-tt
labels: perf, blackhole, matmul
---

## Summary

Qwen3-4B's MLP is `silu(x @ W1) * (x @ W3)` with K = 2560, N = 9728. `ttnn.experimental.minimal_matmul`
with `fuse_swiglu=True` on the packed `[W1 | W3]` weight computes the product in the epilogue, but
the epilogue keeps gate and up pairs in DST, which halves the accumulation subblock and stalls the
FPU while the SFPU does the SiLU. Measured on one Blackhole P150 (12×10, LoFi,
`fp32_dest_acc_en=False`, bfp4_b weights, bfp8_b activations in DRAM, traced):

| M (batch × 512) | unfused FF1 + FF3 + silu·mul | best fused SwiGLU (blocks / subblock) | Δ |
|---|---|---|---|
| 4096 | 1475 µs | 1440 (8,8,8 / 1×8) | −2% |
| 8192 | 3048 | 2874 (4,20,8 / 1×4) | −6% |
| 16384 | 5273 | 5783 (4,8,8 / 1×4) | **+10%** |

The plain FF1 matmul alone runs at 332 TFLOP/s, the product op at 95% of its bandwidth roofline
(1373 µs at M=16384 for 16384×9728 bfp8), and the fused kernel's FLOP rate is 1.62× worse than the
plain matmul. A batched epilogue (several gate/up pairs per DST session) was tried and is slower
(+5.6% at M=4096, +14% at M=16384). In the shipped model the fused kernel is used at M=4096 and
M=8192 and the unfused path at M=16384.

## Why it matters

At M = 16384 the product op is 36 × 1.37 ms ≈ 49 ms and the two matmuls ≈ 177 ms of a 438 ms
forward. A fused kernel that costs "matmul + SiLU on the SFPU" (≈ 4.9 ms per layer) instead of
5.27 ms unfused is ≈ −13 ms; an epilogue that also overlaps the SiLU with the next block's FPU work
would take back most of the 49 ms.

## Ask

An epilogue that does not halve the accumulation subblock: e.g. compute the gate block, pack it
to L1, compute the up block with the full subblock, then multiply in the pack stage or in a short
SFPU pass over the two L1 blocks; or alternate gate/up output blocks per core so the pair never
shares DST. Target: fused ≤ 4.9 ms at M=16384 (≤ unfused − the standalone product op), and no
regression at M=4096/8192.

## Repro / acceptance

Weights are packed as `[W1 | W3]` along N (the kernel's expected layout); the reference is
`silu(x @ W1) * (x @ W3)` in torch (PCC ≥ 0.998 expected, bfp4 weights).

```python
import time, torch, ttnn
B4, B8, K, H = ttnn.bfloat4_b, ttnn.bfloat8_b, 2560, 9728
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)
def cfg(mb, kb, nb, sh, sw):
    return ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                                    compute_with_storage_grid_size=ttnn.CoreCoord(12, 10))
def traced(fn, n=2, reps=4):
    for _ in range(2): [ttnn.deallocate(t) for t in fn()]
    ttnn.synchronize_device(D); tid = ttnn.begin_trace_capture(D, cq_id=0)
    outs = [fn() for _ in range(n)]; ttnn.end_trace_capture(D, tid, cq_id=0)
    ttnn.execute_trace(D, tid, cq_id=0, blocking=True); t0 = time.perf_counter()
    for _ in range(reps): ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / reps / n * 1e6; ttnn.release_trace(D, tid)
    for o in outs:
        for t in o: ttnn.deallocate(t)
    return us
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])
try:
    for M, blk in ((4096, (8, 8, 8, 1, 8)), (8192, (4, 20, 8, 1, 4)), (16384, (4, 8, 8, 1, 4))):
        xt = torch.randn(1, 1, M, K); w1t = torch.randn(1, 1, K, H) * 0.02; w3t = torch.randn(1, 1, K, H) * 0.02
        x = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mk = lambda t: ttnn.from_torch(t, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w1, w3, w13 = mk(w1t), mk(w3t), mk(torch.cat([w1t, w3t], dim=-1))
        c = cfg(8, 8, 8, 1, 8)
        def unfused():
            a = ttnn.experimental.minimal_matmul(x, w1, compute_kernel_config=ckc, config=c, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8)
            b = ttnn.experimental.minimal_matmul(x, w3, compute_kernel_config=ckc, config=c, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8)
            o = ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8)
            ttnn.deallocate(a); ttnn.deallocate(b); return (o,)
        cf = cfg(*blk)
        fused = lambda: (ttnn.experimental.minimal_matmul(x, w13, compute_kernel_config=ckc, config=cf,
                                                          memory_config=ttnn.DRAM_MEMORY_CONFIG, fuse_swiglu=True),)
        xf = ttnn.to_torch(x).float()
        ref = torch.nn.functional.silu(xf @ ttnn.to_torch(w1).float()) * (xf @ ttnn.to_torch(w3).float())
        print(f"M={M:5d}: pcc unfused={pcc(ttnn.to_torch(unfused()[0]), ref):.5f} fused={pcc(ttnn.to_torch(fused()[0]), ref):.5f}")
        u = traced(unfused); f = traced(fused)
        print(f"M={M:5d}: unfused FF1+FF3+mul {u:8.1f} us   fused blk {blk} {f:8.1f} us  ({100*(f/u-1):+5.1f}%)")
        for t in (x, w1, w3, w13): ttnn.deallocate(t)
finally:
    ttnn.close_device(D)
```
