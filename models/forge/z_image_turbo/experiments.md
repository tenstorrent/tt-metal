# DiT Single Pass Performance Optimization Experiments

## System Info
- Device: Blackhole (1x4 mesh, TP=4)
- Core grid: 13x10
- Model: Z-Image-Turbo DiT (6.15B params)
- Resolution: 512x512 (64x64 latent, 1024 image patches, 128 caption tokens)
- Architecture: 2 noise refiners + 2 context refiners + 30 joint transformer blocks

## Results Summary
- **Baseline: 1176.6 ms**
- **Current best: 1005.0 ms (-171.6 ms, -14.6%)**
- **PCC: 0.998256** (threshold: 0.98)

## Active Optimizations (in order of impact)
1. **BF16 RoPE** (was F32 + typecast) — eliminates 408 type conversions/pass (**-83 ms**)
2. **Cached RoPE freqs** — eliminates 204 embeddings + 136 concats/pass (**-48 ms**)
3. **CCL num_links=4** (was 1) — 4x more ethernet links for all-reduce (**-15 ms**)
4. **HiFi2 + packer_l1_acc matmuls** — faster matmul compute config (**-7 ms**)
5. **fp32_dest_acc disabled for norms** — faster norm kernels (**-3 ms**)
6. **Metal trace** — eliminates kernel dispatch overhead (**-3 ms**)
7. **BFP8 MLP weights** — halves DRAM BW for MLP w1/w2/w3 (**-3 ms**)
8. **nlp_create_qkv_heads for RoPE** — faster seq/heads transpose (**-3 ms**)
9. **BFP8 adaLN + embedder** — BFP8 for conditioning weights (**-1 ms**)
10. **Default matmul blocking** (custom was counterproductive) (**-3 ms**)
11. **Fused w1+w3 MLP** via matmul_split (**-0.3 ms**)

## Full Experiment Log (29 experiments)

| # | Avg (ms) | Delta | PCC | Status | Description |
|---|----------|-------|-----|--------|-------------|
| 0 | 1176.6 | - | 1.0000 | baseline | |
| 1 | 1175.2 | -1.4 | 0.9998 | keep→removed | Custom matmul blocking 13x10 |
| 2 | 1170.1 | -6.5 | 0.9997 | **keep** | HiFi2 + packer_l1_acc matmul |
| 3 | 1168.9 | -7.7 | 0.9992 | **keep** | HiFi2 + math_approx REDUCE_KERNEL |
| 4 | 1165.6 | -11.0 | 0.9992 | **keep** | Metal trace capture/replay |
| 5 | 1165.4 | -11.2 | 0.9991 | discard | Disable fp32_dest_acc matmul |
| 6 | 1165.6 | -11.0 | 0.9989 | discard | LoFi math fidelity |
| 7 | 1162.7 | -13.9 | 0.9992 | **keep** | BFP8 MLP weights |
| 8 | 1164.1 | -12.5 | 0.9991 | discard | BFP8 all matmul weights |
| 9 | 1164.6 | -12.0 | 0.9993 | discard | BFP8 MLP + to_out |
| 10 | 1162.5 | -14.1 | 0.9991 | discard | BFP8 MLP + adaLN + embedder |
| 11 | 1162.7 | -13.9 | 0.9992 | discard | l1_small_size 64K |
| 12 | 1162.6 | -14.0 | 0.9992 | discard | SDPA compute kernel |
| 13 | 1160.2 | -16.4 | 0.9992 | **keep** | Remove custom matmul blocking |
| 14 | 1077.1 | -99.5 | 0.9993 | **keep** | **BF16 RoPE** |
| 15 | 1029.1 | -147.5 | 0.9993 | **keep** | **Cache RoPE freqs** |
| 16 | - | - | - | crash | Precompute f_real/f_imag slices |
| 17 | 1029.2 | -147.4 | 0.9993 | discard | trace_region 200MB |
| 18 | 1029.2 | -147.4 | 0.9993 | discard | HiFi2 adaLN matmuls |
| 19 | 1028.8 | -147.8 | 0.9993 | **keep** | Fused w1+w3 MLP |
| 20 | 1028.8 | -147.8 | 0.9992 | discard | BFP8 all matmul natively |
| 21 | 1028.6 | -148.0 | 0.9993 | discard | ttnn.matmul vs minimal_matmul |
| 22 | 1026.0 | -150.6 | 0.9993 | **keep** | nlp_create_qkv_heads for RoPE |
| 23 | 1022.7 | -153.9 | 0.9984 | **keep** | fp32_dest_acc off for norms |
| 24 | 1022.1 | -154.5 | 0.9982 | **keep** | BFP8 adaLN + embedder |
| 25 | 1009.8 | -166.8 | 0.9983 | keep→upgraded | CCL num_links=2 |
| 26 | 1006.9 | -169.7 | 0.9983 | keep→upgraded | CCL num_links=3 |
| 27 | 1005.0 | -171.6 | 0.9983 | **keep** | **CCL num_links=4** |
| 28 | 1005.2 | -171.4 | 0.9633 | discard | BFP4 MLP — PCC below threshold |
| 29 | 1005.6 | -171.0 | 0.9981 | discard | Native MLP BFP8 (no fusion) |

## Key Insights
- **DRAM-bandwidth-bound**: HiFi4→LoFi gives same latency (no compute bottleneck)
- **RoPE was the #1 hidden bottleneck**: F32 + recomputed freqs = 131 ms (11% of baseline)
- **CCL links were under-provisioned**: 1→4 links saved 15 ms
- BFP8 helps MLP weights but NOT attention weights (minimal_matmul doesn't optimize BFP8 reads)
- BFP4 destroys accuracy (PCC 0.963)
- Custom matmul blocking counterproductive with trace

## Plot Data

```csv
experiment,time_ms
0,1176.6
2,1170.1
4,1165.6
7,1162.7
13,1160.2
14,1077.1
15,1029.1
19,1028.8
22,1026.0
23,1022.7
24,1022.1
25,1009.8
26,1006.9
27,1005.0
```
