### Issue
`ttnn.experimental.minimal_matmul` on Blackhole P150 (12×10 grid) reaches very different efficiency on two Qwen3-4B prefill matmuls with identical FLOPs per row. bfp4_b weights (DRAM width-sharded over the 8 banks; interleaved gives the same picture), bfp8_b activations in DRAM, LoFi, `fp32_dest_acc_en=False`, traced:

| M × K × N | config (12×10) | µs | TFLOP/s |
|---|---|---|---|
| 4096 × 2560 × 9728 (FF1/FF3) | blk 8,8,8 sb 1×8 | 615 | **332** |
| 4096 × 9728 × 2560 (FF2) | blk 16,8,8 sb 1×8 | 431 | **473** |
| 8192 × 2560 × 6144 (QKV) | blk 8,8,8 sb 1×8 | 595 | 433 |
| 8192 × 4096 × 2560 (WO) | blk 8,8,8 sb 1×8 | 381 | 451 |

A block sweep (M 4–16, K 4–20, N 4–8, subblocks 1×8 / 2×4 / 1×4) moves the FF1 shape by ≤ 6%; the legacy 2D-multicast kernel is 3–60% slower on these shapes. Likely bound: K = 80 tiles makes the K loop short and in0 (M × K, 10.6 MB) is re-streamed from DRAM once per N block (38 blocks at N = 304 tiles ≈ 400 MB ≈ 650 GB/s at 615 µs, above the DRAM roofline); FF2 re-streams its in0 10 times.

At M = 16384 (batch 32 × 512) FF1 + FF3 are ≈ 177 ms of a 438 ms Qwen3-4B forward; FF2's rate on them is ≈ −53 ms.

### Expected
(K=2560, N=9728) at ≥ 430 TFLOP/s at M = 4096 and 16384 with PCC ≥ 0.999 vs torch, e.g. by keeping an M × K_block in0 slab resident across N blocks or multicasting in0 along the grid row.

### Unit test (random data, one P150, `TT_VISIBLE_DEVICES=<id>`)
Prints µs, TFLOP/s and PCC per shape; passes when the FF1 rows report ≥ 430 TFLOP/s and PCC ≥ 0.999.

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
    ok = True
    for M in (4096, 16384):
        for name, K, N, blk in (("FF1", 2560, 9728, (8, 8, 8, 1, 8)), ("FF2", 9728, 2560, (16, 8, 8, 1, 8))):
            x = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w = sharded_w(torch.randn(1, 1, K, N) * 0.02, K, N)
            mb, kb, nb, sh, sw = blk
            cfg = ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh,
                                           subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(12, 10))
            run = lambda: ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg,
                                                           memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8)
            ref = ttnn.to_torch(x).float() @ ttnn.to_torch(w).float()
            p = pcc(ttnn.to_torch(run()), ref); us = traced(run); tf = 2 * M * K * N / us / 1e6
            print(f"M={M:5d} {name} K={K} N={N}: {us:8.1f} us  {tf:5.0f} TFLOP/s  pcc={p:.5f}")
            if name == "FF1": ok &= tf >= 430 and p >= 0.999
            ttnn.deallocate(x); ttnn.deallocate(w)
    print("PASS" if ok else "FAIL: FF1 shape below 430 TFLOP/s")
finally:
    ttnn.close_device(D)
```
