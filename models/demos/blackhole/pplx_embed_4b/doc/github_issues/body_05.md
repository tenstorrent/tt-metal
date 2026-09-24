### Issue
`MatmulMultiCoreReuseMultiCastProgramConfig` with in1 DRAM **width-sharded** over the 8 P150 banks returns inf / garbage whenever the grid has more in1 columns than banks (9×8 … 12×8), on every shape tried (M = 512; K×N = 2560×6144, 4096×2560, 2560×9728, 9728×2560). Cause in `matmul_multicore_reuse_mcast_2d_program_factory.cpp` (both variants): the per-column DRAM bank walk does
```cpp
worker_core_stride = per_core_N_storage - storage_core_stride;
```
so a column takes the rest of the bank stripe even when its `per_core_N` is smaller than the stripe, and the in1 L1 block is overrun. Capping it,
```cpp
worker_core_stride = std::min(per_core_N_storage - storage_core_stride, per_core_N);
```
makes every wide grid bit-identical to 8×8 (PCC vs torch 0.99992 / 0.99989 / 0.99992 / 0.99974, same as 8×8). Commit `213530ded2a` (branch `arg/pplx-embed-upstream`) carries the fix plus a reader change: `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` (`IN1_DRAM_WIDTH_SHARDED`) issued one NoC read per tile (576 B for bfp4_b, 240 requests per block); the tiles of a block row inside a bank are contiguous in DRAM and in L1, so each row segment is now one multi-burst read (same bytes, same addresses).

M = 512, bfp4_b width-sharded weights, bfp8_b activations in L1, LoFi, traced, one P150:

| K × N | 8×8 per-tile reads (before) | 8×8 row reads | **12×8 row reads** |
|---|---|---|---|
| 2560 × 6144 | 70.3 µs | 67.8 | **49.8** |
| 4096 × 2560 | 51.3 | 49.0 | **40.1** (2×1 subblock) |
| 2560 × 9728 | 114.0 | 110.5 | **79.5** (2×2) |
| 9728 × 2560 | 111.6 | 102.8 | **78.8** (2×1) |

Qwen3-4B batch 1 / 512 tokens end to end: 22.3 → 21.5 ms (reader), 21.4 → 17.7 ms (12×8 grids).

### Expected
Review and upstream both changes (generic for any 2D-multicast matmul with DRAM width-sharded in1) and add a unit test for column count > bank count.

### Unit test (random data, one P150)
Before the fix the 12×8 rows print `finite=False`; after, PCC equals the 8×8 row.

```python
import math, torch, ttnn
B4, B8, T, M = ttnn.bfloat4_b, ttnn.bfloat8_b, 32, 512
D = ttnn.open_device(device_id=0, l1_small_size=32768)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)
banks = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
def pc(gx, bw, sh, sw, pn):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(gx, 8), in0_block_w=bw,
        out_subblock_h=sh, out_subblock_w=sw, per_core_M=2, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=True)
try:
    ok = True
    for K, N, bw in ((2560, 6144, 10), (4096, 2560, 16), (2560, 9728, 10), (9728, 2560, 38)):
        x = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG)
        pad = math.ceil(N / (T * 8)) * (T * 8)
        spec = ttnn.ShardSpec(banks, (K, pad // 8), ttnn.ShardOrientation.ROW_MAJOR)
        w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D,
                            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec))
        ref = ttnn.to_torch(x).float() @ ttnn.to_torch(w).float()
        Nt = N // T; res = {}
        for gx in (8, 12):
            pn = math.ceil(Nt / gx); sw = 2 if pn % 2 == 0 else 1
            o = ttnn.to_torch(ttnn.matmul(x, w, program_config=pc(gx, bw, 2 if gx == 12 else 1, sw, pn),
                                          compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=B8)).float()
            fin = bool(torch.isfinite(o).all())
            res[gx] = float(torch.corrcoef(torch.stack([o.flatten().double(), ref.flatten().double()]))[0, 1]) if fin else float("nan")
            print(f"K={K} N={N} grid {gx}x8 per_core_N={pn}: finite={fin} pcc={res[gx]:.5f}")
        ok &= res[12] == res[12] and abs(res[12] - res[8]) < 1e-4
        ttnn.deallocate(x); ttnn.deallocate(w)
    print("PASS" if ok else "FAIL")
finally:
    ttnn.close_device(D)
```
