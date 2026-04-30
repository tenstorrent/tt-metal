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
- All matmul shapes using default 8x8x8 blocking (no registered configs for 13x10 grid)
- PCC: 1.0000 (reference)

## Key Findings
- 12 unique matmul shapes, none registered for 13x10 core grid
- The 30 joint blocks dominate runtime (1152-wide M dimension)
- Joint block shapes: (1152, 3840, 3072), (1152, 1024, 3840), (1152, 3840, 2560), (1152, 2560, 3840)
- **Model is heavily DRAM-bandwidth-bound**: ~10.8 GB weight reads per forward pass
- Changing math fidelity from HiFi4 to HiFi2 to LoFi gives the same latency, proving compute is not the bottleneck
- Largest gains come from reducing dispatch overhead (trace) and compute config flags (packer_l1_acc)
- Further improvements require L1 sharding or weight compression to reduce DRAM bandwidth pressure

## Experiment Log

| # | Experiment | Avg (ms) | Delta (ms) | PCC | Status |
|---|-----------|----------|------------|-----|--------|
| 0 | Baseline | 1176.6 | 0.0 | 1.0000 | baseline |
| 1 | Register (4,K,4) matmul blocking for 13x10 grid | 1175.2 | -1.4 | 0.9998 | keep |
| 2 | HiFi2 + packer_l1_acc for matmul compute | 1170.1 | -6.5 | 0.9997 | keep |
| 3 | HiFi2 + math_approx_mode for REDUCE_KERNEL | 1168.9 | -7.7 | 0.9992 | keep |
| 4 | Metal trace capture and replay | 1165.6 | -11.0 | 0.9992 | keep |
| 5 | Disable fp32_dest_acc_en for matmul | 1165.4 | -11.2 | 0.9991 | discard |
| 6 | LoFi math fidelity for matmuls | 1165.6 | -11.0 | 0.9989 | discard |

## Current Best: 1165.6 ms (-11.0 ms, -0.93% vs baseline)
- PCC: 0.999243 (well above 0.98 threshold)

### Optimizations Applied
1. Custom matmul blocking for 13x10 grid (better core utilization)
2. HiFi2 + packer_l1_acc for matmul compute kernels
3. HiFi2 + math_approx_mode + packer_l1_acc for norm/activation compute kernels
4. Metal trace capture and replay (eliminates dispatch overhead)

### Future Optimization Opportunities (not attempted due to constraints)
- **L1 sharding** (user constraint: "Do not use L1 sharding yet")
- **Weight compression** (e.g., BFP8 weights to reduce DRAM BW by 2x)
- **Fused matmul + reduce-scatter** (requires model code changes)
- **Async prefetching** of next-block weights during current-block compute

## Plot Data (experiment number vs time in ms)

```csv
experiment,time_ms
0,1176.6
1,1175.2
2,1170.1
3,1168.9
4,1165.6
5,1165.4
6,1165.6
```
