# Round 3 analysis: Timestral-209M, paper shape, one Blackhole p150a (110 worker cores)

Inputs:
- `bench.json` and `bench.log` (final bench, R3-5 code);
- `ops_perf_results.csv`, a reduced profile with embed + 1 `time` layer + 1 `time_variate` layer + the `last` layer + head;
- `perf_report.*`, `sections/*`, `summary.json`, `layer_counts.json`;
- `tracy/reports/2026_10_22_11_40_38/profile_log_device.csv`, the per-core device log;
- `candidates/*` (R3-1, R3-3, R3-4, the settled harness and R3-5) and `qkv_split_sweep_accuracy.*`;
- the code in `tt/` after R3-1, R3-3, R3-4 and R3-5.

Full model = embed + 18 x `layer_time` + 5 x `layer_time_variate` + 1 x `layer_last` + head = **359 device ops** (round 2: 331, round 0: 513). Relative to round 2:
- R3-1 adds 72 fused head-split ops and 24 `nlp_concat_heads`, and removes 24 create-heads and 48 RoPE ops: +24 ops.
- R3-3 adds 3 slice + reshape pairs in the last layer and removes the head's pair: +4 ops.

Conventions are the same as in rounds 0–2:
- "E" = one bf16 activation 98304 x 768 = 151.0 MB. M = 1024 rows x 96 padded tokens (66 real: 64 context + 2 horizon).
- Roofline peaks: HiFi2 359 TFLOPs, 512 GB/s. "FLOPs %" is on the 110-core basis (HiFi2 ≈ 304 TFLOPs) unless marked "/359". tt-perf-report computes FLOPs % against the cores an op uses, so for the 80-core gate matmul it prints 38.6 %; on the 110-core basis that op is at 28.1 %.
- `DRAM %` with `*` is my estimate: logical bytes / 512 GB/s / device time.
  - For the addcmul matmuls (wo, w_down + residual) it includes the 151 MB residual read that tt-perf-report leaves out.
  - For `minimal_matmul_split` it includes all three outputs. tt-perf-report counts only one of them, so it prints 22.8 % for QKV where the estimate is 45.4 %.
- IDs below are tt-perf-report IDs (= row index in `ops_perf_results.csv` + 2).
- CSV quirks:
  - tt-perf-report leaves `MinimalMatmulDeviceOperation` and `FusedRMSNormPostAllGatherDeviceOperation` unclassified ("Other").
  - `summary.json`'s `matmul_weighted_flops_percent` (24.14 %) covers only the 9 2D `MatmulDeviceOperation` rows of the reduced window, where the 4.2 % embed hidden matmul dominates. It is not the matmul utilization of the model (§3).
  - `PROGRAM CACHE HIT` is False on every row. Host issue durations of ≈ 5 µs per op mean the cache does hit, so I treat that column as unreliable.
- Durations are `DEVICE KERNEL DURATION`. The device log has only FW/KERNEL zones per RISC: there are no custom zones and no CB-wait counters. It does give per-core kernel durations, which §6 uses to show load spread inside an op. It cannot localize a stall inside a kernel.

## 1. Headline

| | paper / accuracy | paper / performance | long_context / accuracy | multivariate / accuracy |
|---|---|---|---|---|
| steady wall (median of 5, back-to-back) | **540.7 ms** (runs 526–542) | **501.4 ms** (500–508) | **221.8 ms** (218–223) | **813.9 ms** (804–821) |
| round 2 wall | 679.6 | 688.0 | 274.2 | 975.7 |
| change vs round 2 | −138.9 ms (**−20.4 %**) | −186.6 ms (−27.1 %) | −52.4 ms (−19.1 %) | −161.8 ms (−16.6 %) |
| round 0 wall | 1445.3 | 1353.0 | 463.4 | 1797.5 |
| change vs round 0 | −904.6 ms (**−62.6 %**) | −851.6 ms (−62.9 %) | −241.6 ms (−52.1 %) | −983.6 ms (−54.7 %) |
| device (median of 5, 0.3 s idle before each) | **470.2 ms** | **453.8 ms** | 196.3 ms | 756.2 ms |
| round 2 device (single run, no idle) | 530.9 | 549.9 | 212.7 | 871.9 |
| change vs round 2 | −60.7 ms (**−11.4 %**) | −96.1 ms (−17.5 %) | −16.4 ms (−7.7 %) | −115.7 ms (−13.3 %) |
| round 0 device | 1307.2 | 1251.3 | 394.4 | 1668.9 |
| change vs round 0 | −837.0 ms (**−64.0 %**) | −797.5 ms (−63.7 %) | −198.1 ms (−50.2 %) | −912.7 ms (−54.7 %) |
| overhead (wall − device) / wall | 70.5 ms = **13.0 %** (r2 21.9 %) | 47.6 ms = 9.5 % (r2 20.1 %) | 25.5 ms = 11.5 % (r2 22.4 %) | 57.7 ms = 7.1 % (r2 10.6 %) |
| ops per forward | 359 (r2 331) | 359 | 359 (same graph; not profiled) | not profiled |
| (wall − device) / ops | 0.196 ms/op (r2 0.449) | 0.133 ms/op (r2 0.417) | 0.071 ms/op (r2 0.186) | — |
| prepare / upload / postprocess (medians of 5) | 38.7 / 12.2 / 19.5 ms (sum **70.4**; r2 141.0) | 21.1 / 13.3 / 16.2 ms (sum 50.6; r2 133.1) | 11.8 / 6.1 / 11.7 ms (sum 29.6; r2 59.9) | 22.8 / 12.9 / 17.6 ms (sum 53.3; r2 96.3) |
| unattributed (wall − device − split) | 0.1 ms (r2 7.8) | −2.9 ms | −4.0 ms | 4.3 ms |
| PCC normalized | 0.99994 (r2 0.99993) | 0.99970 (r2 0.99980) | 0.99976 (r2 0.99975) | 0.99994 (r2 0.99993) |
| cold (first call) | 0.70 s (r2 0.90 s) | 0.65 s (r2 0.78 s) | 0.36 s (r2 0.46 s) | 17.0 s (r2 17.0 s) |

- **Harness change.** From R3-5 on, `device_s` is the median of 5 timed forwards, each after a 0.3 s idle. Round 2 took one forward right after the 5 back-to-back steady runs. Wall (`steady_s`) is still back-to-back.
  - Back-to-back device-only forwards at the paper shape slow from ≈ 470 to ≈ 505 ms (`candidates.md`); a 0.3 s idle restores ≈ 471 ms.
  - On the same R3-4 code, the harness change moved paper/accuracy by −6.1 ms (475.5 → 469.4) and left the other three cases within 0.2 ms (performance 453.8 → 453.9, long_context 196.4 → 196.5, multivariate 756.6 → 756.8).
  - So ≈ 6 ms of the −60.7 ms paper/accuracy device change is measurement. The profile-to-profile change (527.86 → 468.43 ms, −59.4 ms, §2) has no harness in it and matches the bench delta.
- **The forward is still device-bound.**
  - The measured device time (470.2 ms) matches the profiled kernel sum (468.4 ms, −0.4 %). In-section op-to-op gaps are ≈ 0.5 µs per op.
  - Under Tracy, the host issues a `time` layer at ≈ 111 µs per op (1.44 ms for 13 ops) while the device takes 18.34 ms. The `last` layer issues at ≈ 180 µs per op. Host issue is ≈ 46 ms per forward and is fully hidden.
  - Wall − device − split is 0.1 ms. Everything else in the wall is the host split.
- **The host split halved:** 141.0 → 70.4 ms at the paper shape (R3-5: per-patch scaler sums and a bf16 cast before `from_torch`).
  - The accuracy prepare runs were 22.1, 39.8, 39.8, 38.7, 37.9 ms. The performance runs, with the same host code, took 19–22 ms.
  - The background `/tmp/eval_full.py` process is still at ≈ 103 % of one CPU (running since Oct 19). That is the most likely cause of the bimodal prepare time, ≈ 18 ms of host noise in the accuracy wall.
- **Performance precision is now faster than accuracy on device:** 453.8 vs 470.2 ms (−16.4 ms); in round 2 it was 19.0 ms slower. R3-4 did this (§8).
- **Multivariate:** multivariate − paper device = **286.0 ms** (round 2: 341.0 ms). That is 57.2 ms per full `time_variate` layer x 5 (round 2: 56.8 ms x 6). R3-3 removed one full layer's worth, and R3-1 gave the common −48.7 ms. The variate attention itself is untouched and still not profiled (F11).
- Per token: long_context runs at 5.81 µs of device time per token (32 x 1056 tokens; round 2: 6.29). Paper runs at 4.78 µs per padded token (round 2: 5.40), which is 6.96 µs per real token (round 2: 7.86).
- long_context gains least (−7.7 % device). R3-3 helps it little because only h = 32 of 1056 tokens are decoder tokens.

## 2. Reconciliation

Estimate from the reduced profile: **468.43 ms** over 359 ops. Measured device: **470.18 ms** (estimate −0.4 %). The profile syncs and places a signpost at every section boundary. tt-perf-report's "tracing could save 2742 µs" comes from those inflated first gaps of each section, which device time does not include. The head's 112–280 µs gaps are the same artifact.

| section | device ms (1 copy) | ops | multiplier | contribution ms | % | round 2 contribution ms |
|---|---|---|---|---|---|---|
| embed | 9.08 | 6 | 1 | 9.08 | 1.9 | 9.29 |
| layer_time | 18.34 | 13 | 18 | 330.05 | 70.5 | 363.43 (20.19 x 18) |
| layer_time_variate | 23.78 | 18 | 5 | 118.89 | 25.4 | 154.40 (25.73 x 6) |
| layer_last (`time_variate` cut to the decoder slots after its time SDPA) | 10.16 | 24 | 1 | 10.16 | 2.2 | — (one of the 6 above) |
| head | 0.25 | 5 | 1 | 0.25 | 0.1 | 0.73 |
| **total** | | **359** | | **468.43** | 100 | 527.86 (331 ops) |

- `layer_time_variate` − `layer_time` = 5.44 ms (round 2: 5.54 ms): the univariate variate sublayer (IDs 106–110: norm 725, wv 1077, wg + sigmoid 1368, MUL 1004, wo + residual 1305 µs = 5.48 ms).
- Per `time` layer: matmuls 11.02 ms (60.1 %), everything else 7.32 ms (39.9 %). Round 2 was 53.9 % / 46.1 %.
- `layer_last` splits into the full-row part (norm, QKV, 3 head splits, SDPA, concat: 8.19 ms), the cut (3 slices + 3 reshapes: 1.43 ms), and everything after it on 2048 rows (0.54 ms).

Where the −59.4 ms (profile to profile) came from:
- R3-1: **−45.1 ms**. The time-attention core (QKV + head split + RoPE + SDPA + concat) went from 9.44 to 7.49 ms per layer (−1.95 ms x 24). The `time` layer total moved −1.86 ms, because the other ops moved +0.09 ms in noise.
- R3-3: **−14.1 ms**. The last layer as a full `time_variate` layer would be 23.78 ms + the round-2 head 0.73 ms; it is now 10.16 + 0.25 ms.
- Embed: −0.2 ms. The embed out matmul is 2413 µs (round 2: 2625 µs); the cause was not investigated.

Round 2 predictions against outcomes:

| change | round 2 estimate | profile outcome | bench outcome (candidate, old harness) |
|---|---|---|---|
| R3-1 fused head split + RoPE (round 2 F1) | −45 to −55 ms (−2.1 to −2.6 ms per layer) | −45.1 ms (−1.95 per layer, QKV +0.6 ms in total) | −36.9 ms (530.9 → 494.0) |
| R3-3 last layer on the decoder slots (F5) | ≈ −14 ms | −14.1 ms | −12.8 ms |
| R3-4 bf16 wo/w_down in performance precision (F4) | ≈ −33 ms, performance only | not profiled | −36.8 ms (490.6 → 453.8) |
| R3-5 host prepare/upload (F3) | −30 to −45 ms wall | — | −59.3 ms wall (601.4 → 542.1), including less host noise |

- R3-1 landed at the low end. The post ops cost 0.80–0.96 ms each (predicted 0.75–1.0) and the concat 0.79 ms (predicted 0.72).
- The candidate benches understate R3-1 by ≈ 8 ms. Under the old harness the measured device time ran 1.5–2.7 % above the profile estimate (494.0 vs ≈ 482.7 ms for R3-1); in round 2 the excess was 0.6 %. That fits the duty-cycle slowdown above. Under the settled harness the excess is 0.2–0.4 %.

## 3. Category split and matmul utilization

- **Full model (sections x multipliers): Compute 81.7 % / TM 18.3 % / Other 0 %**, counting the fused head-split op as TM, which is its main job.
  - Compute 382.9 ms = matmuls 281.4 (60.1 %) + RMSNorm 37.5 (8.0 %) + SDPA 32.7 (7.0 %) + BinaryNg 31.3 (6.7 %).
  - TM 85.5 ms = fused head split (+ RoPE on q/k) 65.1 + `nlp_concat_heads` 19.0 + last-layer slices/reshapes 1.4.
  - Round 2 counted RoPE as compute and create heads as TM (90.1 / 9.9). Under that convention, only the RoPE share of the fused op (≈ 7.4 ms, the q/k excess over v) is compute, which gives 83.3 / 16.7 here.
- Reduced window as printed by tt-perf-report: Other 58.1 / Compute 35.7 / TM 6.2. "Other" is `MinimalMatmulDeviceOperation` (45.0 %) and `FusedRMSNormPostAllGatherDeviceOperation` (13.2 %). Reclassified, it is Compute 80.7 / TM 19.3.
- **Weighted mean matmul FLOPs utilization:**
  - Reduced window, per tt-perf-report (per-op-core basis): 45.84 % for `MinimalMatmulDeviceOperation` (18 ops) and 24.14 % for the 2D `MatmulDeviceOperation` (9 ops: 3 full-size wg, 2 wg on 2048 rows, the embed hidden and 3 head matmuls). Combined: ≈ 41.3 %.
  - **Full model: 43.6 % (304 basis) / 36.9 % (359 basis).** Round 2: 43.9 / 37.2; round 0: 15.6 / 13.2. Flat this round: R3-1 replaced the 2D QKV with a split matmul of the same speed, and R3-3 removed big and small matmuls alike.
  - The three big matmuls (fused SwiGLU, QKV split, w_down + residual): 50.1 % / 42.4 % over 195.0 ms (round 2: 50.6 / 42.9).
  - The 61 768x768 matmuls (wg, wo + residual, wv): 29.2 % / 24.7 % over 79.7 ms (round 2: 29.4 / 24.9).
- All matmuls together do 37.3 TFLOP per forward (round 2: 38.8; R3-3 removed one layer's post-attention matmuls). That is 104.0 ms of math at 359 TFLOPs (122.8 ms at 304), against **281.4 ms** measured (round 2: 290.9 ms).
- Per-op roofline (max of logical DRAM bytes / 512 GB/s and FLOPs / 359 TFLOPs): **294.2 ms** in total, matmuls 147.5 ms and non-matmul 146.7 ms.
  - The matmuls are **1.91x** off their roofline (round 2: 1.89x), a gap of 133.9 ms.
  - The non-matmul ops are 1.27x off theirs, a gap of 40.3 ms.
- Arithmetic intensity is unchanged: the fused SwiGLU (1104 FLOP/B) is the only matmul above the HiFi2 ridge of 701 FLOP/B. QKV split 573, w_down + residual 437, wg/wv 383, wo + residual 255.
- Whole graph: **143.2 GB of logical DRAM traffic per forward** (matmuls 68.1 GB, everything else 75.1 GB; round 2: 153.0 GB). The DRAM floor is **279.7 ms** (round 2: 298.8 ms); the forward runs at 1.67x that floor (round 2: 1.77x).

**Exit criterion (a), main matmuls > 65 % FLOPs AND non-matmul ops < 15 % of device time: NO on both counts.**

| condition | target | measured |
|---|---|---|
| main matmul FLOPs % (304 / 359 basis) | > 65 % | fused SwiGLU 54.3 / 46.0, QKV split 43.8 / 37.0, w_down + residual 50.8 / 43.0 (weighted 50.1 / 42.4); 768x768: wg + sigmoid 28.0 / 23.7, wo + residual 29.5 / 25.0, wv 35.4 / 30.0 |
| non-matmul share of device time | < 15 % | **39.9 %** (187.0 of 468.4 ms; round 2: 44.9 %) |

- At their DRAM floors, wo + residual can reach at most 43 % FLOPs and wg/wv 64.4 %. QKV (96 %), w_down + residual (74 %) and the SwiGLU (compute-bound) could reach 65 %.
- **The non-matmul condition is unreachable with the current op set.** The non-matmul roofline alone is 146.7 ms. If every matmul ran at 65 % (≈ 189 ms in total), the non-matmul ops would have to fit in 33 ms, below a fifth of their own floor. Getting there would need the norms, the head split, SDPA's surroundings and the gate fused into the matmuls.

**Exit criterion (b), round-over-round improvement: device −11.4 % (530.9 → 470.2 ms; −11.3 % profile to profile) and wall −20.4 % (679.6 → 540.7 ms). This is not a plateau.**
- The findings below add up to a realistic **≈ 50 ms of device time (−11 %)** for a round without the split stream: F2, F3, F4, F6, F7 and F8. The split stream (F1) is worth ≈ 80 ms more on its own but is a design study.

## 4. Per-op-type table (full model)

| op type | full-model ms | % | count | round 2 ms | round 2 count | change ms |
|---|---|---|---|---|---|---|
| MinimalMatmulDeviceOperation (incl. split, fused SwiGLU and addcmul) | 239.96 | 51.2 | 110 | 184.74 | 86 | +55.22 |
| MatmulDeviceOperation (2D multicast) | 41.46 | 8.9 | 34 | 106.15 | 58 | −64.69 |
| *all matmuls* | *281.42* | *60.1* | *144* | *290.89* | *144* | ***−9.47*** |
| FusedRMSNormPostAllGatherDeviceOperation (head split, RoPE on q/k) | 65.08 | 13.9 | 72 | 0 | 0 | +65.08 |
| RotaryEmbeddingLlamaDeviceOperation | 0 | 0 | 0 | 78.06 | 48 | **−78.06** |
| NlpCreateHeadsDeviceOperation | 0 | 0 | 0 | 51.77 | 24 | **−51.77** |
| LayerNormDeviceOperation (RMSNorm, HiFi4) | 37.53 | 8.0 | 55 | 38.95 | 55 | −1.42 |
| SDPAOperation | 32.69 | 7.0 | 24 | 34.56 | 24 | −1.87 |
| BinaryNgDeviceOperation (MUL, ADD) | 31.28 | 6.7 | 34 | 33.16 | 34 | −1.88 |
| NLPConcatHeadsDeviceOperation | 18.99 | 4.1 | 24 | 0 | 0 | +18.99 |
| Slice + ReshapeView | 1.43 | 0.3 | 6 | 0.47 | 2 | +0.96 |
| **total** | **468.43** | 100 | **359** | 527.86 | 331 | **−59.43** |

- BinaryNg: 28 full-size gate MULs (28.19 ms), 2 decoder-slot gate MULs in the last layer, 3 embed ops and 1 head ADD.
- All non-matmul ops: 187.0 ms (39.9 %, 215 ops).
- The time-attention core (QKV + head split + SDPA + concat) is now 179.5 ms (38.3 %); in round 2 it was 226.6 ms (QKV + create heads + RoPE + SDPA).

Matmuls by role (the small last-layer calls on 2048 rows are included in their role):

| role | n | round 2 ms | round 3 ms | per call µs (full size) | round 3 engine | change ms |
|---|---|---|---|---|---|---|
| fused SwiGLU | 24 (23 full + 1 small) | 88.70 | 86.30 | 3745 | minimal `fuse_swiglu`, M/K/N 8/12/2, subblock 4x2 | −2.40 |
| QKV | 24 | 62.17 | 62.74 | 2614 | **`minimal_matmul_split`, 3 chunks, M/K/N 16/12/4, subblock 2x4** (round 2: 2D) | +0.57 |
| w_down + residual | 24 (23 + 1) | 47.98 | 46.14 | 2003 | addcmul (minimal 8/8/8) | −1.84 |
| wg + SIGMOID, time + variate | 30 (28 + 2) | 40.76 | 38.24 | 1363 | 2D 8x10 (80 cores) | −2.52 |
| wo + residual, time + variate | 30 (28 + 2) | 38.31 | 36.29 | 1293 | addcmul | −2.02 |
| variate wv | 6 (5 + 1) | 6.47 | 5.42 | 1077 | minimal 8/8/8 | −1.05 |
| embed (3 matmuls) | 3 | 6.28 | 6.06 | — | as round 2 | −0.22 |
| head (3 matmuls) | 3 | 0.21 | 0.21 | — | 2D | 0 |
| **total** | **144** | **290.89** | **281.42** | | | **−9.47** |

- The split QKV (2614 µs) is 22 µs slower per call than round 2's 2D QKV (2592 µs). The sweep (host-timed) had 2.790 ms for the shipped subblock 2x4 and **2.693 ms for 8x1** with the same blocks; see F7.
- All of the −9.5 ms comes from R3-3 cutting the last layer's post-attention matmuls down to 2048 rows.

## 5. Op-to-op map

All ops run on 110 cores except the 80-core gate matmuls and the 64-core norms on 2048 rows. Tensors are DRAM interleaved bf16, matmuls run at HiFi2, and RMSNorm and the fused head-split op run at HiFi4. Configs come from `tt/common.py` → `tt/program_configs.py`:
- "2D" = `MatmulMultiCoreReuseMultiCastProgramConfig` (`in0_block_w=12`, `out_block_h=11`, `per_core_M=308`, `out_subblock_h=1`).
- "minimal" = `ttnn.experimental.minimal_matmul`, M/K/N blocks 8/8/8, subblock 1x8, grid 11x10. "addcmul" = `dit_minimal_matmul_addcmul_fused(x, W, 1.0, residual, ones[1, N])` on that config.
- "split" = `ttnn.experimental.minimal_matmul_split(xn, Wqkv, chunks=3)`, M/K/N 16/12/4, subblock 2x4 (`split_config`, program_configs.py:102).
- "post" = `ttnn.experimental.wan_fused_rmsnorm_post_allgather(t, unit_stats, eps=0, num_heads_per_device=12, ...)`, HiFi4, with the RoPE arguments for q and k only (attention.py:181).
  - `unit_stats` is a constant [1, 1, M, 32] tensor of `d_model / 32` built at upload, so the norm scale is exactly 1.
  - The output [1, 12, M, 64] is viewed as (12, 1024, 96, 64) for SDPA.
- "SwiGLU" = `minimal_matmul(fuse_swiglu=True)`, M/K/N 8/12/2, subblock 4x2.

Bound: tt-perf-report's classification for matmuls. `DRAM*` / `SLOW*` is mine: estimated DRAM % at least 65 % or not.

### One `time` layer (IDs 81–93, 18.34 ms; round 2 20.19 ms; round 0 50.24 ms)

| ID | op | shape / M x K x N | µs | bound | DRAM % | FLOPs % (304 / 359) | config | source |
|---|---|---|---|---|---|---|---|---|
| 81 | rms_norm (attn) | 98304x768 | 724 | DRAM* | 81.5* | — | default, HiFi4 | attention.py:106 → common.py:250 |
| 82 | QKV split | 98304x768x2304 → 3 x 98304x768 | 2616 | SLOW | 45.4* (tpr 22.8) | 43.7 / 37.0 | split 16/12/4, sb 2x4 | attention.py:198 → common.py:213 |
| 83 | post q: head split + RoPE | (1,1,98304,768) → (1,12,98304,64) | 961 | DRAM* (tail) | 67.8* | — | post, HiFi4 | attention.py:200 → :181 |
| 84 | post k: head split + RoPE | same | 958 | DRAM* (tail) | 68.0* | — | post, HiFi4 | attention.py:201 → :181 |
| 85 | post v: head split | same | 802 | DRAM* | 75.1* | — | post, no RoPE | attention.py:202 → :181 |
| 86 | SDPA causal | q/k/v (12,1024,96,64) | 1362 | DRAM* | 86.6* | — | no concat output | attention.py:205 → :156 |
| 87 | nlp_concat_heads | (1,12,98304,64) → (1,1,98304,768) | 789 | DRAM* | 74.8* | — | | attention.py:206 |
| 88 | wg + SIGMOID | 98304x768x768 | 1360 | SLOW | 43.5* | 28.1 / 23.8 (tpr 38.6) | 2D 8x10, sb 1x3 | attention.py:116 → common.py:172 |
| 89 | MUL ctx * gate | 98304x768 | 1007 | DRAM* | 87.9* | — | | attention.py:118 |
| 90 | wo + residual | 98304x768x768 + residual | 1294 | DRAM* | 68.6* | 29.5 / 25.0 | addcmul | attention.py:124 → common.py:183 |
| 91 | rms_norm (MLP) | 98304x768 | 715 | DRAM* | 82.5* | — | default, HiFi4 | mlp.py:43 |
| 92 | fused SwiGLU | 98304x768x4096 → 2048 | 3743 | SLOW | 29.2* | 54.4 / 46.0 | SwiGLU 8/12/2, sb 4x2 | mlp.py:44 → common.py:236 |
| 93 | w_down + residual | 98304x2048x768 + residual | 2004 | DRAM* | 69.0* | 50.8 / 43.0 | addcmul | mlp.py:49 → common.py:183 |

- Post-op bytes per call: q/k read E + the 6.3 MB unit stats + 25.2 MB of cos/sin and write E (333.5 MB); v has no cos/sin (308.3 MB).
- "(tail)": the average DRAM % looks fine, but per-core durations range 273–961 µs (§6).

### One `time_variate` layer (IDs 96–113, 23.78 ms; round 2 25.73 ms; round 0 63.77 ms)

| ID | op | shape / M x K x N | µs | bound | DRAM % | FLOPs % (304 / 359) | source |
|---|---|---|---|---|---|---|---|
| 96 | rms_norm (time attn) | 98304x768 | 724 | DRAM* | 81.5* | — | attention.py:106 |
| 97 | QKV split | 98304x768x2304 → 3 x 768 | 2605 | SLOW | 45.5* | 43.9 / 37.2 | attention.py:198 |
| 98 | post q | (1,1,98304,768) → (1,12,98304,64) | 942 | DRAM* (tail) | 69.1* | — | attention.py:200 |
| 99 | post k | same | 939 | DRAM* (tail) | 69.4* | — | attention.py:201 |
| 100 | post v | same | 800 | DRAM* | 75.3* | — | attention.py:202 |
| 101 | SDPA causal | (12,1024,96,64) | 1363 | DRAM* | 86.5* | — | attention.py:205 |
| 102 | nlp_concat_heads | → (1,1,98304,768) | 800 | DRAM* | 73.7* | — | attention.py:206 |
| 103 | wg + SIGMOID (time) | 98304x768x768 | 1368 | SLOW | 43.3* | 27.9 / 23.6 | attention.py:116 |
| 104 | MUL ctx * gate | 98304x768 | 1008 | DRAM* | 87.8* | — | attention.py:118 |
| 105 | wo (time) + residual | 98304x768x768 + residual | 1278 | DRAM* | 69.4* | 29.9 / 25.3 | attention.py:124 |
| 106 | rms_norm (variate attn) | 98304x768 | 725 | DRAM* | 81.4* | — | attention.py:106 |
| 107 | wv (univariate V-path) | 98304x768x768 | 1077 | SLOW | 55.0* | 35.4 / 30.0 | attention.py:134 |
| 108 | wg + SIGMOID (variate) | 98304x768x768 | 1368 | SLOW | 43.3* | 27.9 / 23.6 | attention.py:116 |
| 109 | MUL v * gate | 98304x768 | 1004 | DRAM* | 88.1* | — | attention.py:118 |
| 110 | wo (variate) + residual | 98304x768x768 + residual | 1305 | DRAM* | 68.0* | 29.2 / 24.8 | attention.py:124 |
| 111 | rms_norm (MLP) | 98304x768 | 721 | DRAM* | 81.8* | — | mlp.py:43 |
| 112 | fused SwiGLU | 98304x768x4096 → 2048 | 3752 | SLOW | 29.2* | 54.2 / 45.9 | mlp.py:44 |
| 113 | w_down + residual | 98304x2048x768 + residual | 1999 | DRAM* | 69.2* | 50.9 / 43.1 | mlp.py:49 |

### The `last` layer (IDs 116–139, 10.16 ms)

| ID | op | shape | µs | note | source |
|---|---|---|---|---|---|
| 116 | rms_norm | 98304x768 | 723 | 81.6 % DRAM* | attention.py:106 |
| 117 | QKV split | 98304x768x2304 | 2623 | 43.6 / 36.9 % FLOPs; q is needed for only 2048 of the 98304 rows | attention.py:198 |
| 118–120 | post q / k / v | (1,12,98304,64) | 947 / 956 / 803 | q on all rows | attention.py:200-202 |
| 121 | SDPA causal | (12,1024,96,64) | 1355 | all 96 query slots | attention.py:205 |
| 122 | nlp_concat_heads | → (1,1,98304,768) | 784 | | attention.py:206 |
| 123 / 124 | slice ctx to decoder slots / reshape copy | (1024, 96→2, 768) → (1,1,2048,768) | 272 / 204 | slice ≈ 72 %, reshape ≈ 51 % DRAM* on the tile-padded 50 MB | attention.py:111 → common.py:261-262 |
| 125 / 126 | slice xn / reshape | same | 270 / 202 | only for the gate (wg) | attention.py:111 |
| 127 / 128 | slice residual x / reshape | same | 277 / 203 | | attention.py:115 |
| 129–131 | wg + SIGMOID (80 cores), MUL, wo + residual | 2048 rows | 46 / 25 / 46 | | attention.py:116-124 |
| 132–136 | variate norm (64 cores), wv, wg, MUL, wo + residual | 2048 rows | 21 / 36 / 46 / 24 / 46 | | attention.py:106-134 |
| 137–139 | MLP norm, SwiGLU, w_down + residual | 2048 rows | 21 / 161 / 67 | | mlp.py:43-49 |

The full-row part is 8.19 ms, the cut 1.43 ms, and the 2048-row tail 0.54 ms. F6 attacks the first two.

Embed (IDs 73–78, 9.08 ms; round 2 9.29 ms):
- 73: hidden 98304x64x3072 + bias + SILU, 3010 µs. 2D with `in0_block_w=2`; 40.0 % DRAM*, 4.2 % FLOPs (residual_block.py:43).
- 74: out 98304x3072x768 + bias, 2413 µs. Minimal; 61.5 % DRAM*, 63.2 / 53.5 % FLOPs, the most efficient matmul in the model (residual_block.py:44).
- 75: residual 98304x64x768 + bias, 639 µs. Minimal (residual_block.py:46).
- 76: ADD main + skip, 1006 µs, 88 % DRAM* (residual_block.py:47).
- 77: MUL keep (broadcast over rows), 1011 µs, ≈ 60 % DRAM* (model.py:235).
- 78: ADD forecast query (broadcast over rows), 1000 µs, ≈ 60 % DRAM* (model.py:237).

Head (IDs 142–146, 0.25 ms): rms_norm on 64 cores 21 µs (model.py:245), then 2D hidden 2048x768x3072 + SILU 103 µs, out 2048x3072x672 77 µs, residual 2048x768x672 30 µs and ADD 24 µs. The round-2 head slice/reshape is gone, because R3-3 does the cut in the last layer.

## 6. Top 10 op roles (full model, with multiplicity)

The lower bound (LB) per call is max(FLOPs / 359 TFLOPs, logical bytes / 512 GB/s), where bytes = inputs (incl. the fused residual, stats and cos/sin) + weight + outputs in bf16. Headroom = (actual − LB) x n.

| # | role | n | full ms | per call µs | M x K x N / shape | cores | DRAM % | FLOPs % (304 / 359) | limit hit | LB/call µs | headroom ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fused SwiGLU 768 → 2x2048 | 23 | 86.1 | 3745 | 98304x768x4096 → 2048 | 110 | 29.2* | 54.3 / 46.0 | neither: per-core time is even (3656–3743 µs), not FPU-bound (round 2), so the per-core pipeline | 1723 (math; DRAM 1094) | **46.5** |
| 2 | fused head split (+ RoPE on q, k) | 72 | 65.1 | 957 / 954 / 801 (q / k / v) | (1,1,98304,768) → (1,12,98304,64) | 110 | 70.2* (q/k 68.2, v 75.1) | — | DRAM on average, but latency-bound per core: 273–961 µs (q) with a position gradient; RoPE adds ≈ 150 µs not hidden | 651 (q/k), 602 (v) | **19.4** |
| 3 | QKV split 768 → 3x768 | 24 | 62.7 | 2614 | 98304x768x2304 | 110 | 45.4* (tpr 22.8) | 43.8 / 37.0 | neither | 1187 (DRAM; math 969) | **34.3** |
| 4 | w_down + residual | 23 | 46.1 | 2003 | 98304x2048x768 + 98304x768 | 110 | 69.0* | 50.8 / 43.0 | DRAM (near) | 1382 (DRAM) | 14.3 |
| 5 | attn gate wg + SIGMOID (2D), 18 + 5 time + 5 variate | 28 | 38.2 | 1363 | 98304x768x768 | **80** | 43.5* | 28.0 / 23.7 | neither: 80 cores + in-kernel sigmoid | 592 (DRAM; math 323) | **21.6** |
| 6 | rms_norm | 52 | 37.5 | 720 | 98304x768 | 110 | 81.9* | — | DRAM, with a position-dependent per-core tail (254–724 µs) | 590 | 6.8 |
| 7 | wo + residual, 23 time + 5 variate | 28 | 36.2 | 1293 | 98304x768x768 + 98304x768 | 110 | 68.6* | 29.5 / 25.0 | DRAM (near) | 887 (DRAM) | 11.4 |
| 8 | SDPA causal on the (H, R, S, Dh) view | 24 | 32.7 | 1362 | q/k/v (12,1024,96,64) | 110 | 86.6* | — | DRAM | 1180 | 4.4 |
| 9 | attn gate MUL (ctx or v) * gate | 28 | 28.2 | 1007 | 98304x768 | 110 | 87.9* | — | DRAM | 885 | 3.4 |
| 10 | nlp_concat_heads | 24 | 19.0 | 791 | (1,12,98304,64) → (1,1,98304,768) | 110 | 74.6* | — | DRAM (near) | 590 | 4.8 |

- Together these ten roles are 451.7 ms (96.4 %), with 166.9 ms of headroom.
- Matmul gaps per call: wg 2.30x LB, QKV 2.20x, SwiGLU 2.17x, wv 1.82x, wo + residual 1.46x, w_down + residual 1.45x.
- Next in line: variate wv (5 x 1077 µs = 5.4 ms, 2.4 ms headroom), embed (9.1 ms, 6 ops), the last-layer cut (1.4 ms, 6 ops), head (0.25 ms).

Per-core spread from the device log (min / median / max of the per-core kernel duration, µs):

| op | min | median | max | pattern |
|---|---|---|---|---|
| rms_norm (81) | 254 | 470 | 723 | column means rise from 304 to 649 µs left to right; the last core row (NOC y = 11) is fast |
| post q (83) | 273 | 601 | 961 | same gradient as rms_norm |
| post v (85) | 136 | 481 | 800 | same gradient |
| nlp_concat_heads (87) | 366 | 706 | 789 | milder |
| SDPA (86) | 922 | 1298 | 1362 | mostly even; the last core row is fast |
| MUL (89) | 805 | 955 | 1007 | even |
| QKV split (82) | 2345 | — | 2616 | smooth gradient along y, no step |
| SwiGLU (92) | 3656 | — | 3743 | even |

- The post op splits its 3072 tile rows evenly (28 per core, contiguous ranges, row-wise core order), and rms_norm does the same. The 2.3–3.5x spread therefore does not come from the work split.
- It is position-dependent, which points at DRAM/NOC latency or contention seen through shallow pipelines. The post-op reader issues 8 tiles and then waits (`async_read_barrier`) per block into a 16-tile input CB. At the start of every tile row it reads the stats, cos and sin, each behind its own barrier. The RMSNorm reader (`reader_unary_interleaved_ln.cpp`) also reads in fixed-size blocks per tile row.
- Ops with deeper or more parallel streams (MUL, SDPA, the matmuls) are even across cores and reach 87–88 % DRAM on similar traffic.
- **q/k cost ≈ 150 µs more than v** (≈ 134 µs even on the fastest cores). Either the HiFi4 RoPE compute is not overlapped with the shallow read pipeline, or the serialized cos/sin barriers add the time. I could not separate the two without in-kernel zones.

Notes on the roles:
- **Fused SwiGLU** (#1): unchanged from round 2. It is not FPU-bound (LoFi gains ≤ 0.4 ms per call), its in0 is re-read 4x, and it has a serialized SFPU epilogue. Per-core times are even, so the limiter is the per-block pipeline. A device-zone profile is the next step (F9).
- **QKV split** (#3): the same speed as the old 2D matmul. The shipped subblock (2x4) is not the sweep's best (8x1, −3.5 %; F7).
- **wg + sigmoid** (#5): still the least efficient matmul (80 cores, 2.30x LB). F3 packs it into the split matmul.
- **Fused head split** (#2), **concat** (#10): R3-1 replaced 129.8 ms of create heads + RoPE with 65.1 + 19.0 = 84.1 ms. What is left is the reader pipeline (F4) and the concat (F2).
- **RMSNorm** (#6), **SDPA** (#8), **gate MUL** (#9): at 82–88 % of DRAM peak. Only fusion or fewer tokens (F1) removes them.

## 7. Findings, ranked by estimated full-model savings

Ranking is by estimated savings weighted by confidence and complexity. Savings overlap and are not additive; the interactions are stated per finding. All estimates are full-model device ms at the paper shape and accuracy precision, unless marked wall, performance-only or multivariate. Findings that need changes to shared ttnn kernels are marked **[ttnn]**. Everything else is model-level, using existing ops.

**F1. Context/decoder split stream. Savings ≈ 60–100 ms on paper, plus ≈ 90 ms on multivariate. Confidence: low. Complexity: high. Model-level, existing ops. Shape-gated: it would cost long_context ≈ +10 ms.**

- Evidence:
  - The reference forecast time mask says "the context never reads the horizon": `enc_dec` is all zeros (`build_forecast_time_mask`, `reference/src/timestral/masks.py:34-46`). The reference's own `build_two_stream_time_mask` (`:68-101`) states the same ("Enc to Dec: never", `:79`). Variate attention is slot-local. Context tokens are therefore independent of decoder tokens in every layer.
  - The 30 pad tokens per row (31.25 % of all work) exist only because 2 decoder tokens push 64 context tokens over a tile boundary.
  - Every op in the layer scales with the padded token count (§5); 98 % of device time is in the 24 layers.
- Mechanism:
  1. **Context stream:** 1024 rows x 64 tokens = 65536 rows (2048 tiles), S = 64 with no padding. It is the current graph at 2/3 of the tokens: 12.2 ms per `time` layer and 15.9 ms per `time_variate` layer.
  2. **Decoder stream:** 2048 rows dense, starting from the forecast query (no embed). Each layer runs its own norm, QKV split and post ops on 2048 rows (≈ 0.3 ms).
  3. **Decoder attention**, using free views of the head-major tensors:
     - decoder q/k/v (1, 12, 2048, 64) as (12, 64, 32, 64): 16 rows x 2 tokens per tile;
     - context K/V (1, 12, 65536, 64) as (12, 64, 1024, 64): 16 rows x 64 tokens.
     - Concatenate K and V on the sequence axis to (12, 64, 1056, 64), then run SDPA with an additive block mask (1, 64, 32, 1056): each decoder query sees its own row's valid context keys and its causal decoder chain.
     - Finish with `nlp_concat_heads` on 2048 rows.
  4. The rest of the decoder layer (gate, wo, variate sublayer, MLP) runs on 2048 rows, as the last layer does today (0.54 ms).
- Cost model per layer:
  - Decoder chain: 2 tile-aligned concats of ≈ 208 MB each, ≈ 0.45 ms each; masked SDPA of ≈ 270 MB, ≈ 0.6 ms; flat ops ≈ 0.53 ms. That is ≈ 2.0 ms, or ≈ 3.5 ms pessimistic.
  - `time` layer 18.34 → 14.2–15.7 ms (−2.6 to −4.1); `time_variate` 23.78 → 18.0–19.5 ms (−4.3 to −5.8).
  - The embed shrinks by a third (−3.0 ms). The last layer only needs context K/V (norm, [Wk | Wv] split and post on 65536 rows): ≈ −3.5 to −4.8 ms.
  - Total: **≈ −75 to −110 ms**, quoted as 60–100 ms because the decoder-chain costs are unmeasured.
- Multivariate: variate attention is slot-local, so the pad slots drop out of it too: ≈ 57.2 ms x 5 x 30/96 ≈ **−90 ms** more (unprofiled).
- long_context (S = 1056 = 33 tiles, no padding) gains ≈ 3 % of its flat work and pays the decoder chain: **≈ +10 ms**. Keep the single stream when the pad fraction is small; gate on roughly ≥ 15 %.
- Risks, all unverified:
  - SDPA with Sq = 32 ≠ Sk = 1056, and a mask that varies over the "heads" dimension on this packed view. Today's time mask already varies over dim 1, so the layout is supported; its speed with Sq ≠ Sk is not measured.
  - Missing patches need a per-row key-validity mask on both streams.
  - Host issue grows by ≈ 12 small ops per layer (≈ 650 ops per forward, ≈ 72 ms host issue), still well under the device time.
  - The numerics are exact up to reduction order.
- Gate: micro-benchmark the decoder attention chain (2 concats + masked SDPA + views) at the paper shape, and proceed if it is ≤ 2 ms per layer. F6 is the same mechanism confined to the last layer and is the cheap way to test it.
- Interactions: F1 uses the head-major (H, R, S, Dh) layout. F2 would change that layout, so under F1, F2 is worth only the context stream's concat (≈ 2/3 of it) and needs a decoder-side relayout. F3 and F4 keep ≈ 2/3 of their value under F1. F1 subsumes F6.

**F2. [ttnn] Batch-capable fused head-split op, so SDPA writes the concat heads and `nlp_concat_heads` goes away. Savings ≈ 15–17 ms (−24 ops). Confidence: medium. A small change to a shared ttnn op.**

- Evidence rows 87, 102, 122: 24 x 791 µs = 19.0 ms at 74.6 % DRAM*. In round 2, SDPA with `output_concat_heads=True` cost 1438 µs against 1362 µs now (+76 µs).
- Mechanism:
  - Today the post op requires batch == 1 and dim 1 == 1 (`TT_FATAL(a.logical_shape()[0] == 1 ...)` in `fused_distributed_rmsnorm/device`). Its writer lays heads out with one global head stride = num_tile_rows x head_dim_tiles. So the model must use the (1, 12, M, 64) → (12, 1024, 96, 64) view, and SDPA's concat-heads output would concatenate the wrong axis.
  - Accept (R, 1, S, D) input and write (R, H, S, Dh): relax the check, use a per-batch head stride in the writer, and index cos/sin by tile_row mod S_t.
  - The last part also lets the model pass (1, 1, S, Dh) cos/sin tables: −25 MB of reads per q/k call, and no `ttnn.repeat` at upload.
  - SDPA on (R, H, S, Dh) with `output_concat_heads=True` (the round-2 path) then writes the flat (R, 1, S, 768) directly. The time mask goes back to (R, 1, S, S).
- Expected: −791 + 76 = **−715 µs x 24 ≈ −17 ms**. A lower bound of ≈ 15 ms allows for the post op writing a less contiguous layout.
- Alternative: an SDPA writer mode that concatenates the batch axis for the (H, R, S, Dh) view. That touches the widely shared SDPA program factory, so it is more invasive.
- Needs the op's unit tests in ttnn. Pick F2 if F1 is not pursued.

**F3. Pack wg into the split QKV matmul, [wv | wg] on the univariate V-path, and apply the sigmoid in the gate MUL (R3-6). Savings ≈ 11–19 ms (−28 ops). Confidence: medium-low. Model-level, existing ops.**

- Evidence rows 82/88, 97/103, 107/108: wg + sigmoid on 80 cores costs 28 x 1363 µs = 38.2 ms at 28 % FLOPs (2.30x LB), the least efficient matmul. The split QKV takes 2616 µs for N = 72 tiles.
- Mechanism:
  - `linear_split(xn, [Wq | Wk | Wv | Wg], p, 4)`: N = 96 tiles, 24 per chunk. That is the chunk width of today's split, so the M16/K12/N4 config should carry over. It returns q, k, v and g_raw.
  - Then `ttnn.mul(merged, g_raw, input_tensor_b_activations=[SIGMOID])`. R2-5 measured the absorbed sigmoid at no extra cost.
  - V-path: `linear_split(xn, [Wv | Wg], p, 2)`.
  - The last layer keeps its 2048-row wg (46 µs).
- Expected:
  - QKVG ≈ 3.27–3.6 ms against 2616 + 1363 = 3979 µs, i.e. −0.38 to −0.71 ms x 23.
    - 3.27 ms assumes the time scales with output tiles per core plus a 25 % fixed part.
    - 3.6 ms assumes worse N-block quantization: 24 blocks of 4 over 10 cores fill 80 % of the slots, against 90 % for today's 18 blocks.
  - V-path ≈ 1.9 ms against 1077 + 1368 µs: ≈ −0.55 ms x 5.
  - Total **−11 to −19 ms**.
- Risk: the split matmul's time at N = 96 is extrapolated. Sweep it with `tests/perf/sweep_qkv_split.py`; many larger blocks already failed CB allocation at N = 72.
- Conflicts with F5's model-only V-path variant (that variant needs the sigmoid gate as a separate tensor).

**F4. [ttnn] Pipeline the fused head-split reader (and the RMSNorm reader). Savings ≈ 8–14 ms (post op) + 2–4 ms (RMSNorm). Confidence: low-medium. Small changes to shared ttnn kernels.**

- Evidence rows 83–85, 98–100, 118–120:
  - q/k at 955 µs (68 % DRAM*) and v at 801 µs (75 %), against 87–88 % for MUL and SDPA on similar traffic.
  - Per-core durations span 273–961 µs with a smooth position gradient while the work split is even (§6).
  - The q/k excess over v (≈ 150 µs) is close to the RoPE time on the fastest cores (134 µs), so the RoPE is not hidden.
- Mechanism:
  - In the reader, issue a full tile row (24 tiles) per barrier into a ≥ 48-tile input CB instead of 8-tile blocks into 16 tiles.
  - Read stats + cos + sin behind one barrier and prefetch the next row's.
  - In the writer, flush per row, not per 8-tile block.
  - The fused op's CBs are tiny (input 16 tiles, cos/sin one head-dim each), so L1 is not the constraint.
- Expected: q/k 955 → 750–800 µs (−7.4 to −9.8 ms over 48 calls) and v 801 → 700–740 µs (−1.4 to −2.4 ms over 24 calls): **≈ 9–12 ms**, quoted as 8–14 ms.
- The same idea in `reader_unary_interleaved_ln.cpp` could bring RMSNorm from 82 % toward 88–90 % DRAM: −50 to −65 µs x 52 ≈ 2.6–3.4 ms.
- Unverified: the cause of the per-core gradient. Add `DeviceZoneScopedN` zones to the reader and compute kernels (tt-profiler skill) before changing anything.

**F5. [ttnn] Fuse the attention gate MUL. Savings ≈ 15–20 ms with a kernel change, ≈ 2.5 ms model-only. Confidence: low.**

- Evidence rows 89, 104, 109: 28 x 1007 µs = 28.2 ms at 88 % DRAM*.
- Kernel options (the gate multiplies the matmul *input*, so no matmul epilogue can absorb it):
  - With today's layout, multiply in `nlp_concat_heads`: it already reads every ctx tile and writes the [M, 768] tile the gate covers.
  - With F2, multiply in SDPA's concat-heads writer.
  - Either way the MUL goes away for ≈ +0.3 ms of gate reads in the host op: ≈ −0.7 ms x 23 ≈ −16 ms.
- Model-only, univariate V-path: `dit_minimal_matmul_addcmul_fused(xn, Wv, 1.0, zeros[M, 768], gate)` computes (xn @ Wv) ⊙ gate. That is ≈ −0.5 ms x 5 = −2.5 ms, with a resident 151 MB zeros tensor. It conflicts with F3 (pick F3).

**F6. Last layer: decoder-only queries. Savings ≈ 3.5–4.5 ms. Confidence: medium-low. Model-level, existing ops. It is also the cheapest test of F1's risks.**

- Evidence rows 116–128: the last layer spends 8.19 ms on full rows and 1.43 ms cutting them. After its SDPA only 2048 of the 98304 query rows matter, but q, its post op, SDPA and concat run on every row, and 3 tensors are cut separately.
- Mechanism:
  1. x_dec = take_slots(x), which the residual needs anyway. xn_dec = rms_norm(x_dec) costs 21 µs.
  2. Run a [Wk | Wv] split on the full rows (≈ 1.75 ms instead of 2.62 ms), then post k and v (1.76 ms).
  3. q = xn_dec @ Wq (≈ 40 µs) + post with the decoder cos/sin (≈ 25 µs), viewed as (12, 64, 32, 64).
  4. K/V viewed as (12, 64, 1536, 64), i.e. 16 rows x 96 slots, then masked SDPA with a (1, 64, 32, 1536) block mask (≈ 0.8 ms), then `nlp_concat_heads` on 2048 rows.
  5. The gate uses xn_dec. Two of the three take_slots disappear.
- Expected: ≈ 6.1–6.6 ms against 10.16 ms → **−3.5 to −4.0 ms**.
- Risks: as for F1 (Sq ≠ Sk SDPA on a packed view with a block mask; per-row key validity when patches are missing).

**F7. Subblock 8x1 for the QKV split. Savings ≈ 2 ms. Confidence: medium. Model-level, a one-line config change.**

- Evidence: in `qkv_split_sweep_accuracy.json`, `split[16,12,4,sb=8x1]` took 2.693 ms and `sb=2x4` took 2.790 ms, with the same PCC (0.999948).
- `split_config` (program_configs.py:102-117) picks the subblock with `_subblock`, which prefers the widest shape on ties (2x4). Its docstring quotes the 2.79 ms number.
- Expected −90 to −100 µs x 24 ≈ **−2.2 ms**, if the host-timed 3.5 % carries over to device time.
- The sweep timed a median of synchronized calls once per config, so repeat it before relying on the 3.5 %.
- Unverified: whether the tall subblock also helps the minimal 8/8/8 configs (wo, w_down, wv: `_subblock` → 1x8) and the SwiGLU (4x2). None of them was swept for subblock shape.

**F8. Embed cleanup. Savings ≈ 1.5–2.5 ms. Confidence: medium. Model-level.**

- Evidence rows 76–78: ADD main + skip 1006 µs, plus the broadcast MUL keep and ADD query at 1011 and 1000 µs (≈ 60 % DRAM*).
- Mechanism:
  - Fold main + skip into the embed out matmul as an addcmul with the skip as the residual (fold the out bias into the skip bias). That costs ≈ +0.25 ms in the matmul for −1.0 ms of ADD.
  - Replace MUL keep + ADD query with one `ttnn.addcmul(add, embedded, keep)`, or with one broadcast ADD of (query − embed(0)) on the decoder and pad slots, since their token inputs are zero.
  - The second form leaves pads at ≈ 0 ± 1 ulp instead of exactly 0. Pads are never valid keys, but this needs the masked PCC tests.

**F9. Matmul internals (SwiGLU, QKV, the addcmul pair) (R3-8). Savings ≈ 1–10 ms. Confidence: low.**

- SwiGLU (46.5 ms headroom, per-core even):
  - `[16,12,2]` measured −1 % (≈ −0.9 ms).
  - `math_approx_mode=True` for the SFPU silu is op-wide, so test the PCC; worth 0–8 ms, unmeasured.
  - Anything else needs a device-zone profile of this op.
- QKV: the R3-1 sweep covered the split matmul (F7 is what is left). wo/w_down + residual are at 69 % DRAM, so ≤ 5 ms remains at an 85 % ceiling, which is kernel-side.

**F10 (performance precision only). bf8 MLP hidden activation. Savings ≈ 9–14 ms of performance-precision device time. Confidence: low-medium.**

- With the SwiGLU output in bf8_b, the 403 MB write and w_down's in0 read halve. w_down + residual is near-DRAM-bound (69 %): −0.4 to −0.6 ms x 23.
- Unverified: whether the addcmul path accepts a bf8 in0 with bf16 weights and residual. Its `TT_FATAL` concerns the residual vs the weight format, not in0.
- The PCC margin is small: performance is at 0.99970 (§8). A bf8 gate (wg + sigmoid output read by the MUL) is ≈ 3–4 ms more, with a similar risk.

**F11 (multivariate only). Variate attention in row-major order with a same-slot mask (R3-10). Savings ≈ 150–250 ms on multivariate (−20 to −33 %). Confidence: low. Profile first.**

- Evidence: multivariate − paper = 286.0 ms of device time (37.8 % of multivariate), 57.2 ms per full `time_variate` layer x 5, unprofiled.
- Mechanism, unchanged from round 2:
  - Rows are ordered r = b·V + v, so the flat tensor is a free view of (B = 128, V·S = 768, D).
  - Use the head split as in the time path, plus a non-causal SDPA with a (1, 1, 768, 768) same-slot additive mask, instead of the two permutes and the 4x padded (12288, 12, 32, 64) SDPA (attention.py:`_variate_context`).
  - ≈ 40–50 ms per layer.
- R3-3 already removed one of the six full layers, hence the lower range than round 2.
- Prerequisite: a reduced multivariate profile.
- Under F1, the context share drops to 2/3 and the decoder share runs on 2048 rows.

**F12. Host wall. Savings ≈ 10–25 ms of wall (no device effect). Confidence: low-medium.**

- Quiet host: ≈ 18 ms of the accuracy wall is prepare-time noise (§1). Re-measure with `/tmp/eval_full.py` stopped before trusting any host-side change.
- Upload (12.2 ms): `unit_stats` (`ttnn.full` of [1, 1, 98304, 32]) and the device-side `ttnn.repeat` of cos/sin are rebuilt on every call (model.py:160-170, :195-204). They depend only on the shape and the positions, so they can be cached per key: ≈ 2–4 ms, unmeasured. F2 would remove the repeat altogether.
- Postprocess (19.5 ms): `monotone_quantiles` + denormalize on the host; vectorizing it or moving it to the device is worth a few ms to ≈ 10 ms. Moving it to the device has an fp32-vs-bf16 precision risk.
- Caution: a faster host raises the device duty cycle in the back-to-back wall loop. Device-only back-to-back forwards run at ≈ 505 ms instead of 470 (§1), so part of a host gain could come back as device time.

### Re-evaluation of the round-2 proposals that were not done this round

| proposal | round 2 estimate | worth now | why |
|---|---|---|---|
| R3-2 RoPE kernel: cos/sin resident across batches | 35–42 ms | **obsolete** | `rotary_embedding_llama` is no longer in the graph. RoPE now runs inside the fused head-split op and costs ≈ 150 µs x 48 ≈ 7 ms over v's time. What is left is that op's reader (F4) and its full-length cos/sin tables (F2). |
| R3-6 pack wg into the split QKV matmul | 12–18 ms | **11–19 ms** (F3) | R3-1 shipped `minimal_matmul_split`, so it is now a model-level change. The wg it replaces is still on 80 cores at 28 % FLOPs. |
| R3-7 unpadded residual stream outside the attention core | 15–40 ms net | **superseded by F1** | F1 removes the padding everywhere, attention included, with free views only. R3-7 needs two row-granular re-layouts of ≈ 255 MB per layer; the slice + reshape pairs here cost ≈ 475 µs per 50 MB. |
| R3-8 matmul gaps: SwiGLU `[16,12,2]`, silu approx mode, QKV re-sweep | 10–20 ms | **1–10 ms** (F9) + ≈ 2 ms (F7) | The QKV re-sweep happened in R3-1. The split is only as fast as the 2D matmul, and the shipped subblock is not the sweep's best. The SwiGLU is even across cores, so a zone profile has to come first. |
| R3-9 RMSNorm program config, embed ADD / MUL cleanup | 5–10 ms | **1.5–2.5 ms model** (F8) + **2–4 ms [ttnn]** (F4) | The RMSNorm limit is the reader pattern (position-dependent per-core tail), not a model-side config. The norm cannot fold into SwiGLU (silu(r·u) ≠ r·silu(u)). The attention norm could move into the post-op stats, but the gate still needs xn, so the net is ≈ 0. |
| R3-10 multivariate variate attention | 200–300 ms on multivariate | **150–250 ms on multivariate** (F11) | R3-3 removed one full variate layer (−67 ms on multivariate). Still unprofiled. |
| Trace capture / dispatch | no device gain, ≤ 8 ms wall | **no device gain, ≈ 0 wall** | Host issue (≈ 46 ms per forward) is hidden behind 470 ms of device time. The unattributed wall residual is 0.1 ms. |
| L1-resident M-chunks | 25–50 ms, very low | **later** | Subsumed in scope by F1 (−1/3 of all traffic). The big remaining round trip is still the SwiGLU → w_down hidden (806 MB per layer). |

Ideas I checked and rejected:
- Folding the attention RMSNorm into the post-op stats (`wan_fused_rmsnorm_pre_allgather(x)` then post on the raw QKV): q/k/v would be fine, since the row scale commutes with the matmul and RoPE. But the gate's sigmoid needs the normalized input, so the norm (or an equivalent full-tensor op) stays: ≈ 0 net.
- Attributing the QKV split's per-core spread to N-block quantization: the spread is a smooth gradient along y (2345 → 2616 µs), not a step.
- Trace capture for the profile's "2742 µs": that is the section-boundary sync artifact, not a real gap.

**Structural floor.** The current graph moves 143.2 GB per forward, a 279.7 ms floor.
- Without F1: F2 + F3 + F4 + F6 + F7 + F8 at their midpoints give ≈ 470 − 16 − 15 − 11 − 4 − 2 − 2 ≈ **420 ms** (≈ 307 ops: F2 −24, F3 −28). Of that, F2 and F4 need ttnn changes; the model-level part alone (F3, F6, F7, F8) gives ≈ 447 ms.
- With F1 instead of F2 and F6, plus F3, F4 and F7 at ≈ 2/3: ≈ 470 − 80 − 10 − 7 − 1.5 ≈ **370 ms**.
- Either way the non-matmul share stays above 35 %.

## 8. Comparison with the performance precision (bf8 weights, LoFi)

- Device: 470.2 → 453.8 ms (**−16.4 ms, −3.5 %, faster**); in round 2 it was +19.0 ms slower.
- Wall: 540.7 → 501.4 ms (−39.3 ms). ≈ 18 ms of that is the quieter prepare in the performance runs (21.1 vs 38.7 ms for the same host code), so the like-for-like wall gap is ≈ 21 ms.
- PCC normalized: 0.99970 (round 2: 0.99980), against 0.99994 for accuracy.

History of the gap on the paper shape (candidate benches; old harness up to R3-4):

| step | accuracy device | performance device | performance − accuracy | performance PCC |
|---|---|---|---|---|
| round 2 final | 530.9 | 549.9 | +19.0 | 0.99980 |
| + R3-1 fused head split + RoPE | 494.0 | 505.9 | +11.9 | 0.99978 |
| + R3-3 last layer on decoder slots | 481.2 | 490.6 | +9.4 | 0.99978 |
| + R3-4 bf16 wo/w_down in performance | 475.5 (same graph) | 453.8 | **−21.7** | **0.99970** |
| settled harness (R3-4 code) | 469.4 | 453.9 | −15.5 | 0.99970 |
| final bench (R3-5 code) | 470.2 | 453.8 | −16.4 | 0.99970 |

- **R3-4 did what round 2's F4 predicted:** −36.8 ms in performance precision (predicted ≈ −33 ms). The wo/w_down residual add now runs in the addcmul epilogue under LoFi with bf16 weights, which removes 54 standalone ADDs. bf8 weights bought nothing on these DRAM-bound ops.
- **The PCC cost of R3-4 is 0.8e-4** (0.99978 → 0.99970), still 0.7e-3 above a 0.999 gate. The cause is not isolated: LoFi with bf16 instead of bf8 weights in wo/w_down, or the epilogue add order. I did not run a per-layer PCC.
- R3-1 and R3-3 also narrowed the gap (+19.0 → +9.4 ms): the ops they removed (create heads, RoPE, the last layer's tail) were precision-independent, or cheaper under LoFi in performance mode. Each candidate step carries ≈ ±5 ms of old-harness noise, so I do not attribute the remaining few ms further.
- **Harness sensitivity differs by precision.** On the same R3-4 code, the settled harness changed accuracy by −6.1 ms and performance by +0.1 ms; long_context and multivariate (both accuracy) did not move either. This fits a power-related clock reduction under sustained HiFi2 matmul load, since LoFi draws less. It is a hypothesis only: no AICLK, power or temperature telemetry was captured.
- **Precision-independent time:** the fused head split (HiFi4 in both modes, because RoPE must stay exact), RMSNorm (HiFi4), SDPA, BinaryNg, concat and slices: 187.0 ms = 39.9 % of accuracy device time (round 2: 44.9 %).
- **What would make performance precision faster:**
  - bf8 MLP hidden (F10): ≈ 9–14 ms.
  - bf8 gate output: ≈ 3–4 ms.
  - Both are activation-precision changes with PCC risk at a 0.99970 starting point. LoFi alone has little left to give: no remaining op is clearly FPU-bound (the SwiGLU's LoFi gain is ≤ 0.4 ms per call).
- A performance-precision profile was again not captured. The per-op attribution above is inferred from the candidate deltas and the HiFi2 profile.
