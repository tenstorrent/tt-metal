### Issue
Qwen3-4B's MLP is `silu(x @ W1) * (x @ W3)` (K = 2560, N = 9728). `minimal_matmul(fuse_swiglu=True)` on the packed `[W1 | W3]` weight computes the product in the epilogue, but the epilogue holds gate/up pairs in DST, halving the accumulation subblock and stalling the FPU while the SFPU runs the SiLU. P150, 12×10, LoFi, `fp32_dest_acc_en=False`, bfp4_b weights, bfp8_b activations in DRAM, traced:

| M (batch × 512) | unfused FF1 + FF3 + silu·mul | best fused (blocks / subblock) | Δ |
|---|---|---|---|
| 4096 | 1475 µs | 1440 (8,8,8 / 1×8) | −2% |
| 8192 | 3048 | 2874 (4,20,8 / 1×4) | −6% |
| 16384 | 5273 | 5783 (4,8,8 / 1×4) | **+10%** |

The plain FF1 matmul runs at 332 TFLOP/s and the product op at 95% of its bandwidth roofline (1373 µs at M = 16384), yet the fused kernel's FLOP rate is 1.62× worse than the plain matmul; a batched epilogue (several pairs per DST session) is slower still (+5.6% / +14%). At M = 16384 the product op is 36 × 1.37 ≈ 49 ms and the two matmuls ≈ 177 ms of a 438 ms forward.

### Expected
An epilogue that keeps the full accumulation subblock (e.g. pack the gate block to L1, compute the up block at full subblock, multiply in the pack stage or a short SFPU pass; or alternate gate/up output blocks per core so a pair never shares DST). Target: fused ≤ 4.9 ms at M = 16384 (≤ unfused minus the standalone product op) with no regression at M = 4096 / 8192, PCC ≥ 0.998 vs torch.

### Unit test (random data, one P150)
Weights packed `[W1 | W3]` along N; reference `silu(x @ W1) * (x @ W3)` in torch. Passes when fused ≤ unfused at every M and PCC ≥ 0.998.

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
    ok = True
    for M, blk in ((4096, (8, 8, 8, 1, 8)), (8192, (4, 20, 8, 1, 4)), (16384, (4, 8, 8, 1, 4))):
        w1t, w3t = torch.randn(1, 1, K, H) * 0.02, torch.randn(1, 1, K, H) * 0.02
        x = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        pf = pcc(ttnn.to_torch(fused()[0]), ref)
        u, f = traced(unfused), traced(fused)
        print(f"M={M:5d}: unfused {u:8.1f} us   fused {f:8.1f} us ({100*(f/u-1):+5.1f}%)  pcc fused={pf:.5f}")
        ok &= f <= u and pf >= 0.998
        for t in (x, w1, w3, w13): ttnn.deallocate(t)
    print("PASS" if ok else "FAIL")
finally:
    ttnn.close_device(D)
```
