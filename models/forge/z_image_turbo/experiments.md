# DiT Single Pass Performance Optimization Experiments

## System Info
- Device: Blackhole (1x4 mesh, TP=4)
- Core grid: 13x10
- Model: Z-Image-Turbo DiT (6.15B params)
- Resolution: 512x512 (64x64 latent, 1024 image patches, 128 caption tokens)
- Architecture: 2 noise refiners + 2 context refiners + 30 joint transformer blocks

## Baseline
- **Avg latency: 1176.6 ms**

## Current Best: 1029.1 ms (-147.5 ms, -12.5%)
- PCC: 0.999282 (well above 0.98 threshold)

## Experiment Log (18 experiments)

| # | Experiment | Avg (ms) | Delta | PCC | Status |
|---|-----------|----------|-------|-----|--------|
| 0 | Baseline | 1176.6 | - | 1.0000 | baseline |
| 1 | Custom (4,K,4) matmul blocking 13x10 | 1175.2 | -1.4 | 0.9998 | keep (later removed) |
| 2 | HiFi2 + packer_l1_acc matmul | 1170.1 | -6.5 | 0.9997 | keep |
| 3 | HiFi2 + math_approx REDUCE_KERNEL | 1168.9 | -7.7 | 0.9992 | keep |
| 4 | Metal trace capture/replay | 1165.6 | -11.0 | 0.9992 | keep |
| 5 | Disable fp32_dest_acc_en | 1165.4 | -11.2 | 0.9991 | discard |
| 6 | LoFi math fidelity | 1165.6 | -11.0 | 0.9989 | discard |
| 7 | BFP8 MLP weights | 1162.7 | -13.9 | 0.9992 | keep |
| 8 | BFP8 all matmul weights | 1164.1 | -12.5 | 0.9991 | discard |
| 9 | BFP8 MLP + to_out | 1164.6 | -12.0 | 0.9993 | discard |
| 10 | BFP8 MLP + adaLN + embedder | 1162.5 | -14.1 | 0.9991 | discard |
| 11 | l1_small_size 32K→64K | 1162.7 | -13.9 | 0.9992 | discard |
| 12 | SDPA compute kernel config | 1162.6 | -14.0 | 0.9992 | discard |
| 13 | Remove custom matmul blocking | 1160.2 | -16.4 | 0.9992 | keep |
| 14 | **BF16 RoPE (was F32)** | **1077.1** | **-99.5** | 0.9993 | **keep** |
| 15 | **Cache RoPE freqs** | **1029.1** | **-147.5** | 0.9993 | **keep** |
| 16 | Precompute f_real/f_imag slices | - | - | - | crash |
| 17 | trace_region_size 70M→200M | 1029.2 | -147.4 | 0.9993 | discard |
| 18 | HiFi2 adaLN matmuls | 1029.2 | -147.4 | 0.9993 | discard |

## Active Optimizations (in order of impact)
1. **Cached RoPE frequency tables** — eliminates 204 embeddings + 136 concats/pass (-48 ms)
2. **BF16 RoPE operations** — was F32 + typecast, saves 408 type conversions/pass (-83 ms)
3. **HiFi2 + packer_l1_acc for matmuls** — faster compute config (-6.5 ms)
4. **HiFi2 + math_approx for norms** — faster norm + activation kernels (-1.2 ms)
5. **Metal trace** — eliminates kernel dispatch overhead (-3.3 ms)
6. **BFP8 MLP weights** — halves DRAM bandwidth for MLP (-2.9 ms)
7. **Default matmul blocking** — custom blocking was counterproductive with trace (-2.5 ms)

## Key Insights
- Model is DRAM-bandwidth-bound: HiFi4→LoFi gives same latency
- **RoPE was the hidden bottleneck** — F32 ops + recomputed freqs = 131 ms (11% of baseline)
- BFP8 helps MLP weights but NOT attention weights (regression)
- Custom matmul blocking was counterproductive when combined with trace
- Compute config changes give diminishing returns (<1% total)

## Remaining Bottleneck Analysis
At 1029 ms, runtime is dominated by:
- Matmul weight reads: ~2.7 GB × 4 devices ÷ ~10 GB/s/device ≈ 1080 ms theoretical
- The remaining ~50 ms headroom comes from CCL, element-wise ops, and SDPA
- Further gains require L1 sharding or deeper architectural changes

## Plot Data

```csv
experiment,time_ms
0,1176.6
1,1175.2
2,1170.1
3,1168.9
4,1165.6
7,1162.7
13,1160.2
14,1077.1
15,1029.1
```
