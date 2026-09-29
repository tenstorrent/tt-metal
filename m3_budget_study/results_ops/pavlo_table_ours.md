# Pavlo's per-op table under our conditions ([2,4] MoE/MSA layer)

Source: `per_op.csv`, which holds the P0-A zone profiles. Each run is one (2,4) stage with SP=2, TP=4 and EP=8
(16 experts per chip), running layers 0-6 contiguously so routing is real. The runs use a 1D fabric, v1
dispatch/combine, bf4 experts, `M3_MOE_W_NDSHARD=1`, `M3_MOE_HYBRID_THRESHOLD=128`, a bf8 index_k cache, a real
prefix (`PROFILE_PREFIX_QUIET=1`) and a warm point. Pavlo's reference is `pavlo_reference.md`: one 5120-token
chunk over 51,200 cached tokens.

## How the columns are computed

* **h.** The 141,312-token request is rounded down to whole chunks, so the profiled chunk attends 139,264 cached
  tokens. That gives kv_len = 143,360 at W=4096 and 147,456 at W=8192.
* **worst / mean / min ms.** For each chip we sum the op's zones and take the max, mean and min over the 8 chips.
  These come from `per_device.json`, then are averaged over sparse layers 3-6. The code column gives the worst ms
  for the code input at the same point.
* **roofline ms.** This is `sim_core` `roofTok`/`roofSeg` via `tools/roofline_ops.js --segments W:139264 --idx bf8`.
  Our index_k cache is bf8 (`BFLOAT8_B` in the `ag_index_k` op). `per_op.csv`'s `roof_ms` for `ag_idx` assumed
  bf16, which overstates it 1.88x; everything else matches `per_op.csv`. The experts roofline carries imbalance
  1.2 with Tr = W, and norm_ag covers both all-gathers.
* **eff (mean, Pavlo form).** This is `zoneEff`: `roof / (mean ms - floor)`, with a floor of 0.04 ms for CCL ops
  (x2 for norm_ag) and 0.01 ms for the rest, clamped to [0.003, 1]. It is directly comparable to Pavlo's eff,
  which is also a mean over chips.
* **eff (worst chip).** `roof / worst ms`, with no floor.
* **target / headroom.** The target is 70% for matmul-class ops and 80% for DRAM- and link-class ops. Headroom is
  `target / min(eff_mean, target)`, the formula of Pavlo's table.
* **share (worst chip).** The op's worst-chip ms divided by the sparse layer's worst-chip ms. The worst chips
  differ from op to op, so a column sums to more than 100% (the MoE chain is the reason; see
  `moe_chain_summary.md`). Pavlo's share is mean over mean.

## [2,4] MoE/MSA layer, single request

### W = 4096, h = 139,264, single prose request (sparse layers 3-6 averaged)

Sparse layer: worst chip 14.40 ms, chip mean 14.21 ms (code input: 13.58 ms), sum of rooflines 3.97 ms. Pavlo's layer (5120 tok, 51k cached): 18.58 ms.

| op | worst ms | mean ms | code worst ms | roofline ms | eff (mean, Pavlo form) | Pavlo eff | eff (worst chip) | target | headroom | Pavlo headroom | share (worst chip) | Pavlo share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 1.120 | 0.998 | 1.114 | 0.377 | 41.1% | 41.7% | 33.7% | 80% | x1.9 | x1.9 | 7.8% | 6.5% |
| qkv | 0.251 | 0.250 | 0.251 | 0.191 | 79.6% | 87.6% | 76.0% | 70% | x1.0 | x1.0 | 1.8% | 1.5% |
| idx_branch | 0.402 | 0.400 | 0.403 | 0.013 | 3.4% | 3.8% | 3.3% | 70% | x20.6 | x18.5 | 2.8% | 2.4% |
| misc | 0.916 | 0.893 | 0.910 | 0.098 | 11.1% | 11.2% | 10.7% | 80% | x7.2 | x7.2 | 6.4% | 6.0% |
| o_proj | 0.312 | 0.310 | 0.312 | 0.170 | 56.5% | 58.5% | 54.4% | 70% | x1.2 | x1.2 | 2.2% | 2.0% |
| attn_rs | 0.531 | 0.485 | 0.517 | 0.189 | 42.4% | 41.6% | 35.5% | 80% | x1.9 | x1.9 | 3.7% | 3.3% |
| shared | 0.972 | 0.948 | 0.975 | 0.379 | 41.8% | 42.0% | 39.1% | 70% | x1.7 | x1.7 | 6.8% | 6.3% |
| router | 0.188 | 0.187 | 0.189 | 0.011 | 6.0% | 6.7% | 5.6% | 70% | x11.7 | x10.4 | 1.3% | 1.1% |
| dispatch | 1.138 | 0.838 | 0.911 | 0.126 | 15.8% | 14.7% | 11.1% | 80% | x5.1 | x5.4 | 7.8% | 6.0% |
| experts | 2.759 | 1.865 | 2.378 | 1.453 | 78.3% | 44.7% | 52.7% | 70% | x1.0 | x1.6 | 19.1% | 18.9% |
| combine | 2.022 | 0.828 | 1.160 | 0.252 | 31.9% | 29.5% | 12.4% | 80% | x2.5 | x2.7 | 13.9% | 6.0% |
| moe_reduce | 2.652 | 2.011 | 1.894 | 0.287 | 14.6% | 11.3% | 10.8% | 80% | x5.5 | x7.1 | 18.2% | 17.3% |
| ag_kv | 0.490 | 0.456 | 0.545 | 0.195 | 46.8% | 39.3% | 39.8% | 80% | x1.7 | x2.0 | 3.4% | 1.3% |
| ag_idx | 0.224 | 0.220 | 0.224 | 0.097 | 54.2% | 100.0% | 43.5% | 80% | x1.5 | x1.0 | 1.6% | 0.5% |
| indexer | 0.570 | 0.568 | 0.569 | 0.022 | 3.9% | 7.8% | 3.8% | 70% | x17.9 | x9.0 | 4.0% | 1.5% |
| sparse | 2.994 | 2.952 | 2.993 | 0.113 | 3.8% | 3.9% | 3.8% | 70% | x18.2 | x17.9 | 20.8% | 19.5% |

### W = 8192, h = 139,264, single prose request (sparse layers 3-6 averaged)

Sparse layer: worst chip 27.00 ms, chip mean 26.54 ms (code input: 25.33 ms), sum of rooflines 6.67 ms. Pavlo's layer (5120 tok, 51k cached): 18.58 ms.

| op | worst ms | mean ms | code worst ms | roofline ms | eff (mean, Pavlo form) | Pavlo eff | eff (worst chip) | target | headroom | Pavlo headroom | share (worst chip) | Pavlo share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 2.169 | 1.951 | 2.151 | 0.755 | 40.3% | 41.7% | 34.8% | 80% | x2.0 | x1.9 | 8.1% | 6.5% |
| qkv | 0.448 | 0.447 | 0.449 | 0.381 | 87.3% | 87.6% | 85.1% | 70% | x1.0 | x1.0 | 1.7% | 1.5% |
| idx_branch | 0.753 | 0.745 | 0.751 | 0.026 | 3.6% | 3.8% | 3.5% | 70% | x19.4 | x18.5 | 2.8% | 2.4% |
| misc | 1.797 | 1.733 | 1.770 | 0.197 | 11.4% | 11.2% | 10.9% | 80% | x7.0 | x7.2 | 6.7% | 6.0% |
| o_proj | 0.590 | 0.586 | 0.589 | 0.339 | 58.8% | 58.5% | 57.5% | 70% | x1.2 | x1.2 | 2.2% | 2.0% |
| attn_rs | 0.996 | 0.914 | 1.028 | 0.377 | 43.2% | 41.6% | 37.9% | 80% | x1.9 | x1.9 | 3.7% | 3.3% |
| shared | 1.829 | 1.785 | 1.821 | 0.759 | 43.5% | 42.0% | 41.5% | 70% | x1.6 | x1.7 | 6.8% | 6.3% |
| router | 0.353 | 0.349 | 0.354 | 0.021 | 6.2% | 6.7% | 6.0% | 70% | x11.2 | x10.4 | 1.3% | 1.1% |
| dispatch | 2.276 | 1.682 | 1.762 | 0.252 | 15.3% | 14.7% | 11.1% | 80% | x5.2 | x5.4 | 8.4% | 6.0% |
| experts | 4.237 | 2.563 | 3.529 | 1.911 | 74.8% | 44.7% | 45.1% | 70% | x1.0 | x1.6 | 15.6% | 18.9% |
| combine | 3.808 | 1.535 | 2.240 | 0.503 | 33.7% | 29.5% | 13.2% | 80% | x2.4 | x2.7 | 13.9% | 6.0% |
| moe_reduce | 5.123 | 3.910 | 3.659 | 0.574 | 14.8% | 11.3% | 11.2% | 80% | x5.4 | x7.1 | 18.7% | 17.3% |
| ag_kv | 0.831 | 0.628 | 0.860 | 0.201 | 34.1% | 39.3% | 24.1% | 80% | x2.3 | x2.0 | 3.1% | 1.3% |
| ag_idx | 0.232 | 0.226 | 0.230 | 0.100 | 53.9% | 100.0% | 43.2% | 80% | x1.5 | x1.0 | 0.9% | 0.5% |
| indexer | 1.232 | 1.191 | 1.231 | 0.044 | 3.7% | 7.8% | 3.6% | 70% | x18.7 | x9.0 | 4.6% | 1.5% |
| sparse | 6.369 | 6.297 | 6.371 | 0.226 | 3.6% | 3.9% | 3.5% | 70% | x19.5 | x17.9 | 23.6% | 19.5% |

**Reading the single-request tables**

* **Token-proportional ops agree with Pavlo within a few points.** This covers norm_ag, qkv, o_proj, attn_rs,
  shared, router, idx_branch, misc and sparse. The mesh and collectives are the same as his [2,4], and each op's
  time scales with W the way its roofline does.
* **The MoE ops are where we differ:**
  * `experts` runs at 78% of its mean-chip roofline against Pavlo's 45%, because of the hybrid and ND-shard
    paths.
  * The worst chip is 1.5x (W=4096) to 1.65x (W=8192) the mean.
  * `combine` is 2.4-2.5x worse on its worst chip than on the mean, and `moe_reduce` 1.3x. Both are waits (see
    below).
* **The depth-dependent ops cost more at 139k than at Pavlo's 51k.** `ag_kv` is 0.49 ms against 0.235 ms and
  `ag_idx` 0.22 ms against 0.09 ms. `indexer` is 0.57 ms against 0.285 ms. It runs at 3.9% against Pavlo's 7.8%
  only because his roofline grows with kv_len/128 while the kernel grows with kv_len (see `msa_summary.md`).
* **`sparse` is the single largest op.** It takes 21-24% of the worst-chip layer and does not depend on depth:
  2.99 ms at 139k and 2.94 ms at 549k. It depends only on rows per chip.

## Packed forwards

### Packed forward, W = 4096 (2 x 2048: prose@141,312 + code@0)

Sparse layer: worst chip 14.47 ms, chip mean 14.36 ms, sum of rooflines 3.97 ms.

| op | worst ms | mean ms | roofline ms | eff (mean, Pavlo form) | share (worst chip) | worst vs single prose |
|---|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 1.076 | 0.981 | 0.377 | 41.9% | 7.4% | -0.044 |
| qkv | 0.251 | 0.249 | 0.191 | 79.6% | 1.7% | +0.000 |
| idx_branch | 0.455 | 0.453 | 0.013 | 3.0% | 3.1% | +0.053 |
| misc | 2.452 | 1.817 | 0.098 | 5.4% | 17.0% | +1.537 |
| o_proj | 0.313 | 0.311 | 0.170 | 56.3% | 2.2% | +0.002 |
| attn_rs | 0.489 | 0.470 | 0.189 | 43.9% | 3.4% | -0.043 |
| shared | 0.977 | 0.951 | 0.379 | 41.7% | 6.8% | +0.005 |
| router | 0.188 | 0.186 | 0.011 | 6.0% | 1.3% | -0.000 |
| dispatch | 0.964 | 0.803 | 0.126 | 16.5% | 6.7% | -0.173 |
| experts | 2.387 | 1.964 | 1.453 | 74.4% | 16.5% | -0.371 |
| combine | 1.255 | 0.653 | 0.252 | 41.1% | 8.6% | -0.767 |
| moe_reduce | 1.808 | 1.362 | 0.287 | 21.7% | 12.4% | -0.844 |
| ag_kv | 0.570 | 0.511 | 0.198 | 42.0% | 3.9% | +0.080 |
| ag_idx | 0.247 | 0.237 | 0.099 | 50.2% | 1.7% | +0.023 |
| indexer | 0.371 | 0.369 | 0.018 | 5.0% | 2.6% | -0.199 |
| sparse | 3.679 | 3.043 | 0.113 | 3.7% | 25.4% | +0.685 |

### Packed forward, W = 8192 (4 x 2048: prose@548,864 + code@141,312 + prose@16,384 + code@0)

Sparse layer: worst chip 29.74 ms, chip mean 29.44 ms, sum of rooflines 7.83 ms.

| op | worst ms | mean ms | roofline ms | eff (mean, Pavlo form) | share (worst chip) | worst vs single prose |
|---|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 2.129 | 1.940 | 0.755 | 40.6% | 7.2% | -0.039 |
| qkv | 0.449 | 0.447 | 0.381 | 87.2% | 1.5% | +0.001 |
| idx_branch | 0.909 | 0.907 | 0.026 | 3.0% | 3.1% | +0.157 |
| misc | 3.662 | 2.980 | 0.197 | 6.6% | 12.3% | +1.865 |
| o_proj | 0.591 | 0.587 | 0.339 | 58.7% | 2.0% | +0.000 |
| attn_rs | 0.993 | 0.923 | 0.377 | 42.7% | 3.3% | -0.003 |
| shared | 1.822 | 1.787 | 0.759 | 43.4% | 6.1% | -0.006 |
| router | 0.353 | 0.349 | 0.021 | 6.2% | 1.2% | +0.000 |
| dispatch | 1.812 | 1.569 | 0.252 | 16.5% | 6.1% | -0.463 |
| experts | 3.342 | 2.612 | 1.911 | 73.4% | 11.2% | -0.896 |
| combine | 2.374 | 1.233 | 0.503 | 42.2% | 8.0% | -1.434 |
| moe_reduce | 3.222 | 2.398 | 0.574 | 24.3% | 10.8% | -1.901 |
| ag_kv | 2.310 | 2.232 | 0.972 | 44.3% | 7.8% | +1.479 |
| ag_idx | 1.112 | 1.100 | 0.486 | 45.9% | 3.7% | +0.880 |
| indexer | 2.235 | 2.228 | 0.048 | 2.1% | 7.5% | +1.003 |
| sparse | 6.835 | 6.151 | 0.226 | 3.7% | 23.0% | +0.466 |

**Reading the packed tables**

* **Packing narrows the MoE skew.** Load max/mean falls from 2.1 to 1.6 (`moe_chain_summary.md`).
  * At W=4096, `experts`, `combine` and `moe_reduce` drop by 0.37, 0.77 and 0.84 ms on the worst chip against
    single prose.
  * At W=8192 they drop by 0.90, 1.43 and 1.90 ms.
* **Attention is per-segment, so its cost goes up:**
  * `idx_branch`, `ag_*`, `indexer` and `sparse` each run one MSA chain per segment. The W=8192 forward has one
    segment at 549k, so its ag_kv, ag_idx and indexer are dominated by that deep segment.
  * The h=0 segment takes the no-cache MSA path with bf16 K/V (Anomaly B in `msa_summary.md`). At W=4096 its
    `sparse_sdpa_msa` costs 2.04 ms on SP rank 1 against 0.81 ms on rank 0. The 2048-row segment at 141k costs
    1.44-1.51 ms.
* **`misc` grows by 1.5-1.9 ms**, for two reasons:
  * **A wait.** The MoE routing-setup all-gather on axis 0 takes 1.25 ms (W=4096) and 1.34 ms (W=8192) on the
    SP-rank-0 chips. They are waiting for rank 1's slower h=0 segment.
  * **Real per-segment work.** Each segment adds its own slice, concat, RoPE, KV write and head split/concat: 12
    slices per layer at B=2 and 24 at B=4, against 4 for a single request. This is the P1-F batching input.

## Ranked op list: headroom ms x share

**How it is computed.** All values are per op, on single prose at h = 139,264, averaged over sparse layers 3-6.

* `headroom ms = max(0, worst ms - roof ms / target)`, using Pavlo's targets: 70% for matmul-class ops and 80%
  for DRAM- and link-class ops. The latency floor is not added to the target time.
* The rank key is `headroom ms x share`, where share is the op's worst-chip share of the sparse layer.
* **Min-chip headroom ms** is the same formula applied to the fastest chip. It is what is left once the
  cross-chip waits are removed. For combine and moe_reduce the gap between the two headroom columns is expert-load
  imbalance, not kernel time.
* **Packed headroom ms** comes from the packed forward at the same W.

### Ranked, W = 4096 (single prose, h = 139,264)

| rank | op | worst ms | min-chip ms | roof / target ms | headroom ms | share | headroom ms x share | min-chip headroom ms | packed headroom ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | sparse | 2.994 | 2.911 | 0.161 | 2.833 | 20.8% | **0.590** | 2.749 | 3.518 |
| 2 | moe_reduce | 2.652 | 0.828 | 0.359 | 2.293 | 18.2% | **0.417** | 0.469 | 1.449 |
| 3 | combine | 2.022 | 0.377 | 0.315 | 1.708 | 13.9% | **0.238** | 0.062 | 0.941 |
| 4 | experts | 2.759 | 1.444 | 2.076 | 0.683 | 19.1% | **0.131** | 0.000 | 0.311 |
| 5 | dispatch | 1.138 | 0.719 | 0.157 | 0.980 | 7.8% | **0.077** | 0.562 | 0.807 |
| 6 | misc | 0.916 | 0.879 | 0.123 | 0.793 | 6.4% | **0.050** | 0.756 | 2.330 |
| 7 | norm_ag (x2) | 1.120 | 0.917 | 0.472 | 0.648 | 7.8% | **0.050** | 0.445 | 0.605 |
| 8 | shared | 0.972 | 0.926 | 0.542 | 0.429 | 6.8% | **0.029** | 0.384 | 0.435 |
| 9 | indexer | 0.570 | 0.566 | 0.031 | 0.539 | 4.0% | **0.021** | 0.535 | 0.345 |
| 10 | attn_rs | 0.531 | 0.448 | 0.236 | 0.295 | 3.7% | **0.011** | 0.212 | 0.253 |
| 11 | idx_branch | 0.402 | 0.397 | 0.019 | 0.383 | 2.8% | **0.011** | 0.378 | 0.436 |
| 12 | ag_kv | 0.490 | 0.434 | 0.244 | 0.246 | 3.4% | **0.008** | 0.190 | 0.323 |
| 13 | router | 0.188 | 0.185 | 0.015 | 0.173 | 1.3% | **0.002** | 0.170 | 0.172 |
| 14 | ag_idx | 0.224 | 0.216 | 0.122 | 0.102 | 1.6% | **0.002** | 0.094 | 0.123 |
| 15 | o_proj | 0.312 | 0.309 | 0.242 | 0.070 | 2.2% | **0.002** | 0.066 | 0.071 |
| 16 | qkv | 0.251 | 0.248 | 0.272 | 0.000 | 1.8% | **0.000** | 0.000 | 0.000 |

### Ranked, W = 8192 (single prose, h = 139,264)

| rank | op | worst ms | min-chip ms | roof / target ms | headroom ms | share | headroom ms x share | min-chip headroom ms | packed headroom ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | sparse | 6.369 | 6.195 | 0.323 | 6.046 | 23.6% | **1.430** | 5.872 | 6.512 |
| 2 | moe_reduce | 5.123 | 1.643 | 0.718 | 4.405 | 18.7% | **0.825** | 0.925 | 2.504 |
| 3 | combine | 3.808 | 0.650 | 0.629 | 3.178 | 13.9% | **0.442** | 0.020 | 1.745 |
| 4 | experts | 4.237 | 1.764 | 2.730 | 1.508 | 15.6% | **0.236** | 0.000 | 0.612 |
| 5 | dispatch | 2.276 | 1.426 | 0.315 | 1.961 | 8.4% | **0.164** | 1.112 | 1.498 |
| 6 | misc | 1.797 | 1.690 | 0.246 | 1.551 | 6.7% | **0.103** | 1.445 | 3.416 |
| 7 | norm_ag (x2) | 2.169 | 1.817 | 0.944 | 1.225 | 8.1% | **0.099** | 0.873 | 1.186 |
| 8 | indexer | 1.232 | 1.151 | 0.063 | 1.169 | 4.6% | **0.053** | 1.088 | 2.167 |
| 9 | shared | 1.829 | 1.753 | 1.084 | 0.744 | 6.8% | **0.051** | 0.669 | 0.738 |
| 10 | idx_branch | 0.753 | 0.739 | 0.038 | 0.715 | 2.8% | **0.020** | 0.701 | 0.871 |
| 11 | attn_rs | 0.996 | 0.865 | 0.472 | 0.524 | 3.7% | **0.019** | 0.394 | 0.521 |
| 12 | ag_kv | 0.831 | 0.446 | 0.251 | 0.580 | 3.1% | **0.018** | 0.195 | 1.095 |
| 13 | router | 0.353 | 0.346 | 0.030 | 0.323 | 1.3% | **0.004** | 0.316 | 0.323 |
| 14 | o_proj | 0.590 | 0.583 | 0.484 | 0.106 | 2.2% | **0.002** | 0.099 | 0.106 |
| 15 | ag_idx | 0.232 | 0.223 | 0.125 | 0.107 | 0.9% | **0.001** | 0.097 | 0.505 |
| 16 | qkv | 0.448 | 0.446 | 0.545 | 0.000 | 1.7% | **0.000** | 0.000 | 0.000 |

**What the ranking means**

1. **sparse.** It ranks first at both W and has no imbalance component (its min chip is within 3% of its worst).
   The kernel is DRAM-gather-bound: it refetches K/V for every query token at about 400 GB/s. Pavlo's roofline
   assumes 32x K/V reuse. A kernel fork for M3 is needed (see `msa_summary.md`).
2. **moe_reduce and combine.** Ranks 2 and 3 are about 80% (moe_reduce) and 96% (combine) cross-chip wait on
   the hot expert's column.
   * The real kernel headroom is 0.47 and 0.06 ms at W=4096.
   * moe_reduce's kernel headroom is mostly `post_combine_reduce` reading all 4 top-k slots. Only about 1 in 4
     is local.
   * The rest of the headroom belongs to routing imbalance. See `moe_chain_summary.md` for the attribution.
3. **experts.** On the mean chip it is already at or above target. Its worst-chip headroom is imbalance.
4. **dispatch, misc, norm_ag, shared and indexer** are real kernel headroom of 0.4-0.8 ms each at W=4096.
   * dispatch has a small imbalance part (worst 1.14 ms against min 0.72 ms).
   * misc more than doubles when packing (per-segment ops plus the h=0 SP wait).
   * indexer's headroom against Pavlo's roofline is largely a roofline artifact. Against its real FLOPs it runs at
     about 26% of LoFi peak.

## Dense GQA layer (layer 1, single prose, h = 139,264)

The ring roofline is `ring_c`, the compute part. With a kv_len-bounded capacity, ring_scan is 0.19-0.20 ms, well
under the compute part. The Pavlo columns are his [2,4] dense layer (layer 0, chunk 5120, 51k).

**W = 4096** (layer worst 26.71 ms)

| op | worst ms | mean ms | roofline ms | eff (mean, Pavlo form) | Pavlo eff | target | headroom | share (worst chip) | Pavlo share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 1.151 | 0.993 | 0.377 | 41.3% | 43.3% | 80% | x1.9 | 4.3% | 7.4% |
| qkv | 0.251 | 0.249 | 0.191 | 79.7% | 87.7% | 70% | x1.0 | 0.9% | 1.8% |
| misc | 0.754 | 0.747 | 0.098 | 13.3% | 13.1% | 80% | x6.0 | 2.8% | 6.0% |
| o_proj | 0.309 | 0.307 | 0.170 | 57.0% | 58.8% | 70% | x1.2 | 1.2% | 2.4% |
| attn_rs | 1.019 | 0.635 | 0.189 | 31.7% | 41.7% | 80% | x2.5 | 3.8% | 3.9% |
| dense_mlp | 1.683 | 1.632 | 0.952 | 59.8% | 61.1% | 70% | x1.2 | 6.3% | 12.6% |
| ring (ring_c) | 21.736 | 21.722 | 7.799 | 35.9% | 35.8% | 70% | x1.9 | 81.4% | 65.9% |

**W = 8192** (layer worst 55.12 ms)

| op | worst ms | mean ms | roofline ms | eff (mean, Pavlo form) | Pavlo eff | target | headroom | share (worst chip) | Pavlo share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 2.252 | 1.985 | 0.755 | 39.6% | 43.3% | 80% | x2.0 | 4.1% | 7.4% |
| qkv | 0.448 | 0.447 | 0.381 | 87.4% | 87.7% | 70% | x1.0 | 0.8% | 1.8% |
| misc | 1.490 | 1.460 | 0.197 | 13.6% | 13.1% | 80% | x5.9 | 2.7% | 6.0% |
| o_proj | 0.580 | 0.578 | 0.339 | 59.7% | 58.8% | 70% | x1.2 | 1.1% | 2.4% |
| attn_rs | 2.747 | 1.470 | 0.377 | 26.4% | 41.7% | 80% | x3.0 | 5.0% | 3.9% |
| dense_mlp | 3.238 | 3.127 | 1.903 | 61.7% | 61.1% | 70% | x1.1 | 5.9% | 12.6% |
| ring (ring_c) | 44.810 | 44.748 | 15.824 | 35.4% | 35.8% | 70% | x2.0 | 81.3% | 65.9% |

The dense `attn_rs` worst chip (1.0 / 2.7 ms against a 0.44-0.46 ms floor) is a wait. It is the first TP
collective after the ring-joint SDPA, and it absorbs the chips' ring skew.

## What differs from Pavlo's calibration, and why

1. **Shape: 5120 tokens at 51,200 cached, against 4096 / 8192 at 139,264.**
   * Token-proportional ops scale with W and keep their efficiency.
     * qkv is 79.6% at 4096 and 87.3% at 8192, against Pavlo's 87.6% at 5120. The weight read is a larger part at
       4096.
     * norm_ag, attn_rs, shared, o_proj, misc and sparse are within ±2 points.
   * Depth-proportional ops grow with kv_len.
     * **indexer.** Its efficiency halves (7.8% to 3.9%). The kernel's work is `2·rows·kv_len·128` (the full
       index rectangle), while the sim roofline is `2·rows·4·128·(2048 + kv_len/128)`. The roofline grows 32x
       slower with depth. At 549k we measure 1.90 ms, about 2.3% of that roofline.
     * **ag_idx.** Pavlo clamped it to 100% because at 51k his roofline sat above the measured value minus the
       0.04 ms floor. At 139k it is measurable at 54%.
     * **ag_kv.** It moves the whole persistent gather buffer, about capacity rows. It measures 47% at W=4096 and
       34% at W=8192. Its own time is about 0.45 ms (layers 4-6). Layer 3, the first gather after the dense
       layer-2 ring, adds an SP arrival wait: 1.6 ms on the SP-rank-0 chips at W=8192.
2. **SP=2 on both sides.** Pavlo's [2,4] is the same mesh as ours: SP=2 on axis 0, TP=4 on axis 1, the 1D fabric
   and Linear CCLs. The TP collectives (norm_ag, attn_rs, the shared reduce-scatter) therefore reproduce his to
   within 1-2 points, and so do the dense-layer ring (35.9% against 35.8%) and dense_mlp. Nothing in our table is
   an SP effect except the arrival waits:
   * **h=0.** Here SP rank 1 does all the causal work and rank 0 waits. This is Anomaly B.
   * **Dispatch and combine.** They run on axis 0 (the SP pair of a column), so a hot chip stalls its SP partner.
3. **ND-sharded weights and hybrid experts (`M3_MOE_W_NDSHARD=1`, `M3_MOE_HYBRID_THRESHOLD=128`).**
   * experts is at 76.9% / 74.5% (mean chip, W=4096 / 8192, prose and code pooled) against Pavlo's 44.7%,
     against the same read-plus-compute roofline with imbalance 1.2.
   * The bench fit (`bench/experts_fit.txt`) says the deployed hybrid path is still additive: t = a·W + b·T, R²
     0.990 against 0.986 for max. Its effective weight bandwidth is 427 GB/s and its token cost is 0.274 µs,
     about 413 TFLOP/s. That is why it now beats a sum-roofline built on 512 GB/s and 608 TF.
   * On the worst chip experts is 53% / 45%: the gap is load imbalance.
4. **Real routing, measured on the worst chip.** Pavlo's efficiencies are means over the 8 chips, and so are ours
   in the eff column and the JSON. Real M3 routing on prose puts max/mean chip load at 2.1, on code 1.65 and on a
   packed forward 1.5-1.6 (`load_skew.csv`). Against our own table that shows up three ways:
   * **combine and moe_reduce means are lower than his, but still hold wait time.** The mean includes the waits of
     7 chips. combine is 31.9% against 29.5%, moe_reduce 14.6% against 11.3%.
   * **combine's worst chip is 2.4x its mean.** Pavlo's own note says 2.4 as well.
   * **The worst-chip efficiencies of the MoE chain are 11-13%.** The kernels (min chip) are close to the bench:
     moe_reduce min 0.83 ms against 0.85 ms for the bench fused+RS at 2048 tokens per chip.
5. **bf8 index_k cache.** Pavlo profiled with `M3_INDEX_CACHE_BF16=1`. Ours is bf8, so ag_idx's roofline uses
   1.0625 B.
6. **The h=0 path.** The first chunk of a request (cached_len = 0) runs a different MSA path: gathered bf16 K/V
   and no cache read. Pavlo's calibration never sees it. It costs 5.25 ms of sparse at W=4096, against 2.99 ms at
   139k (`msa_summary.md`, Anomaly B).

## Our CAL.effs['2x4']

The files, all in the same form as `tools/cal_effs_calibrated.json`:

* `tools/cal_effs_ours_2x4_w4096.json`
* `tools/cal_effs_ours_2x4_w8192.json`
* `tools/cal_effs_ours_2x4.json`, which is the W=4096 set.

Each file is `{"about": [...], "2x4": {"moe": {...}, "dense": {...}}}`, loadable with
`run_goodput.js --effs FILE` or `roofline_ops.js --detail --effs FILE`. The sim keeps one set per mesh, so pass the
file that matches the batching budget.

**Choices, following `sim_core.zoneEff`:**

* **Statistic.** zone ms is the mean over the 8 chips, as in zoneEff and Pavlo's calib. The worst-chip per-op
  times do not sum to the layer, because the waits sit on different chips. The chip means do sum to it.
* **Condition.** Single request at h = 139,264. Prose and code are pooled: moe is the mean of layers 3-6, dense is
  layer 1.
* **Formula.** `roof / (mean - floor)`, clamped to [0.003, 1]. The floor is 0.04 ms for CCL ops (x2 for norm_ag)
  and 0.01 ms otherwise. misc is the layer minus the mapped ops, and kv_a2a is set equal to dispatch. ring_scan is
  Pavlo's placeholder, 0.015.
* **Roofline.** `roofline_ops.js --segments W:139264 --idx bf8`, with imbalance 1.2 and Tr = W.
* **Wave factor.** sparse, indexer and ring_c are multiplied by `waveFactor(W)/waveFactor(5120)`: 1.0417 at 4096
  and 0.9896 at 8192. The sim multiplies by the inverse at run time, so `layerMs` reproduces the measured layer.
* **Check.** `roofline_ops.js --detail --effs <file> --segments W:139264 --idx bf8` gives a sparse layer of
  13.809 / 25.719 ms and a dense layer of 26.241 / 53.821 ms. These equal the measured chip-mean layers exactly.

**Caveats:**

* With `opEff = 0` the sim uses `min(eff, target)`, so qkv and experts (above target at both W) have no effect beyond it.
* For [2,4] pipelines the dense ring is driven by `CAL.pipe.ringC`, not `dense.ring_c`.
* The experts efficiency is a mean-chip number. The imbalance the sim applies is the fixed 1.2 in the roofline,
  against a measured max/mean of 1.5-2.1. The MoE-chain imbalance is spread over combine, moe_reduce and misc in
  the means.

| kind | op | W=4096 mean ms | W=4096 eff | W=8192 mean ms | W=8192 eff | Pavlo (calibrated) |
|---|---|---:|---:|---:|---:|---:|
| moe | norm_ag | 1.002 | 0.4095 | 1.962 | 0.4012 | 0.4173 |
| moe | qkv | 0.250 | 0.7960 | 0.447 | 0.8731 | 0.8756 |
| moe | idx_branch | 0.400 | 0.0339 | 0.745 | 0.0360 | 0.0378 |
| moe | o_proj | 0.310 | 0.5649 | 0.586 | 0.5884 | 0.5846 |
| moe | attn_rs | 0.483 | 0.4265 | 0.919 | 0.4296 | 0.4158 |
| moe | shared | 0.947 | 0.4186 | 1.785 | 0.4348 | 0.4201 |
| moe | router | 0.187 | 0.0600 | 0.349 | 0.0625 | 0.0673 |
| moe | dispatch | 0.807 | 0.1641 | 1.608 | 0.1605 | 0.1473 |
| moe | experts | 1.900 | 0.7689 | 2.576 | 0.7445 | 0.4469 |
| moe | combine | 0.703 | 0.3795 | 1.319 | 0.3934 | 0.2952 |
| moe | moe_reduce | 1.726 | 0.1703 | 3.343 | 0.1738 | 0.1132 |
| moe | ag_kv | 0.470 | 0.4539 | 0.635 | 0.3371 | 0.3934 |
| moe | ag_idx | 0.220 | 0.5411 | 0.226 | 0.5390 | 1.0000 |
| moe | indexer | 0.568 | 0.0408 | 1.191 | 0.0370 | 0.0781 |
| moe | sparse | 2.948 | 0.0401 | 6.298 | 0.0356 | 0.0392 |
| moe | misc | 0.891 | 0.1116 | 1.728 | 0.1144 | 0.1117 |
| dense | norm_ag | 0.995 | 0.4125 | 1.989 | 0.3954 | 0.4328 |
| dense | qkv | 0.249 | 0.7982 | 0.447 | 0.8734 | 0.8768 |
| dense | o_proj | 0.307 | 0.5699 | 0.578 | 0.5969 | 0.5883 |
| dense | attn_rs | 0.593 | 0.3414 | 1.474 | 0.2633 | 0.4166 |
| dense | dense_mlp | 1.629 | 0.5991 | 3.125 | 0.6169 | 0.6111 |
| dense | misc | 0.746 | 0.1336 | 1.460 | 0.1356 | 0.1305 |
| dense | ring_c | 21.722 | 0.3742 | 44.748 | 0.3500 | 0.3576 |
