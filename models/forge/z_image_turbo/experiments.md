# DiT Single Pass Performance Optimization Experiments

## System Info
- Device: Blackhole (1x4 mesh, TP=4)
- Core grid: 13x10
- Model: Z-Image-Turbo DiT (6.15B params)
- Resolution: 512x512 (64x64 latent, 1024 image patches, 128 caption tokens)
- Architecture: 2 noise refiners + 2 context refiners + 30 joint transformer blocks

## Baseline
- **Avg latency: 1176.6 ms**
- Compute kernel: HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
- All matmul shapes using default 8x8x8 blocking
- RoPE: F32 complex rotation with typecast back to BF16
- RoPE freqs: recomputed every call (204 embedding lookups + 136 concats per pass)
- PCC: 1.0000 (reference)

## Experiment Log

| # | Experiment | Avg (ms) | Delta | PCC | Status |
|---|-----------|----------|-------|-----|--------|
| 0 | Baseline | 1176.6 | - | 1.0000 | baseline |
| 1 | Register (4,K,4) matmul blocking for 13x10 | 1175.2 | -1.4 | 0.9998 | keep (later removed) |
| 2 | HiFi2 + packer_l1_acc for matmul | 1170.1 | -6.5 | 0.9997 | keep |
| 3 | HiFi2 + math_approx for REDUCE_KERNEL | 1168.9 | -7.7 | 0.9992 | keep |
| 4 | Metal trace capture and replay | 1165.6 | -11.0 | 0.9992 | keep |
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
| 16 | Precompute f_real/f_imag slices | - | - | - | crash (trace incompatible) |

## Current Best: 1029.1 ms (-147.5 ms, -12.5% vs baseline)
- PCC: 0.999282 (well above 0.98 threshold)

### Active Optimizations
1. HiFi2 + packer_l1_acc for matmul compute kernels
2. HiFi2 + math_approx_mode + packer_l1_acc for norm/activation compute kernels
3. BFP8 (BFLOAT8_B) for MLP weights (102 weights, halves MLP DRAM BW)
4. Metal trace capture and replay (eliminates dispatch overhead)
5. **BF16 RoPE operations** (was F32 + typecast, saves 408 ops/pass) — biggest single win
6. **Cached RoPE frequency tables** (eliminates 204 embeddings + 136 concats/pass)

### Key Insights
- Model is DRAM-bandwidth-bound: HiFi4→HiFi2→LoFi all give same latency
- BFP8 helps MLP weights (-2.9 ms) but not attention weights (regression)
- **RoPE was the hidden bottleneck**: F32 ops + frequent recomputation of constant freqs accounted for ~130 ms (11% of baseline)
- Removing custom matmul blocking improved perf with trace — default 8x8x8 is better

### Future Opportunities (not attempted)
- L1 sharding (user constraint)
- Fused matmul + reduce-scatter
- Async weight prefetching
- Pre-sliced RoPE f_real/f_imag (crashed with trace, needs investigation)

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
