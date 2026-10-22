# Round 2 analysis: Timestral-209M, paper shape, one Blackhole p150a (110 worker cores)

Inputs: `bench.json`, `ops_perf_results.csv` (reduced profile: embed + 1 `time` layer + 1 `time_variate` layer + head), `perf_report.*`, `sections/*`, `summary.json`, `candidates/*` (R2-1 … R2-5 benches), `swiglu_sweep_accuracy.md`, `rope_microbench.md`, and the code in `tt/` after R2-4, R2-1, R2-3 and R2-2.
Full model = embed + 18 x layer_time + 6 x layer_time_variate + head = **331 device ops** (round 1: 457, round 0: 513). The 126 ops removed are 54 residual ADDs (now in the wo / w_down epilogue), 24 MLP MULs and 24 MLP matmuls (fused SwiGLU), and 24 concat-heads ops (now written by SDPA).
Conventions are the same as rounds 0 and 1:
- "E" = one bf16 activation 98304 x 768 = 151.0 MB. M = 1024 rows x 96 padded tokens (66 real).
- Roofline peaks: HiFi2 359 TFLOPs, 512 GB/s. "FLOPs %" is on the 110-core basis (HiFi2 ≈ 304 TFLOPs) unless marked "/359". tt-perf-report computes FLOPs % against the cores an op uses, so for the 80-core gate matmul it prints 38.6 %; on the 110-core basis that op is at 28.1 %.
- `DRAM %` with `*` is my estimate: logical bytes / 512 GB/s / device time. For the addcmul matmuls (wo, w_down + residual), tt-perf-report leaves out the 151 MB residual read: it prints 46.5 % / 54.4 %, while the value including the residual is 69.7 % / 69.1 %. I use the latter.
- IDs below are tt-perf-report IDs (= row index in `ops_perf_results.csv` + 2).
- CSV quirks: on program-cache hits, the input-shape columns of Binary rows are stale, so elementwise shapes come from the code. `dit_minimal_matmul_addcmul_fused` is logged as `MinimalMatmulDeviceOperation`. The fused SwiGLU is logged as `98304 x 768 x 4096` (the [silu | mul] weight width); its output is 98304 x 2048.
- Durations are `DEVICE KERNEL DURATION`. This profile has no per-RISC CB-wait counters: `DEVICE COMPUTE CB WAIT FRONT` is empty, and every RISC duration equals the kernel duration. So the profile cannot localize stalls inside an op.

## 1. Headline

| | paper / accuracy | paper / performance | long_context / accuracy | multivariate / accuracy |
|---|---|---|---|---|
| steady wall (median of 5) | **679.6 ms** (runs 634–686) | **688.0 ms** (runs 647–726) | **274.2 ms** (273–282) | **975.7 ms** (971–982) |
| round 1 wall | 836.4 | 817.0 | 345.9 | 1209.3 |
| change vs round 1 | −156.8 ms (**−18.7 %**) | −129.0 ms (−15.8 %) | −71.7 ms (−20.7 %) | −233.6 ms (−19.3 %) |
| round 0 wall | 1445.3 | 1353.0 | 463.4 | 1797.5 |
| change vs round 0 | −765.7 ms (−53.0 %) | −665.0 ms (−49.1 %) | −189.2 ms (−40.8 %) | −821.8 ms (−45.7 %) |
| device (`forward_device` + sync) | **530.9 ms** | **549.9 ms** | 212.7 ms | 871.9 ms |
| round 1 device | 702.0 | 670.2 | 272.9 | 1059.3 |
| change vs round 1 | −171.1 ms (**−24.4 %**) | −120.3 ms (−17.9 %) | −60.2 ms (−22.1 %) | −187.4 ms (−17.7 %) |
| round 0 device | 1307.2 | 1251.3 | 394.4 | 1668.9 |
| change vs round 0 | −776.3 ms (**−59.4 %**) | −701.4 ms (−56.1 %) | −181.7 ms (−46.1 %) | −797.0 ms (−47.8 %) |
| overhead (wall − device) / wall | 148.7 ms = **21.9 %** (r1 16.1 %) | 138.1 ms = 20.1 % (r1 18.0 %) | 61.6 ms = 22.4 % (r1 21.1 %) | 103.8 ms = 10.6 % (r1 12.4 %) |
| ops per forward | 331 (r1 457) | 331 | 331 (same graph) | not profiled |
| (wall − device) / ops | 0.449 ms/op (r1 0.294) | 0.417 ms/op (r1 0.321) | 0.186 ms/op (r1 0.160) | — |
| prepare / upload / postprocess | 73.1 / 48.9 / 19.0 ms (sum **141.0**; r1 105.5) | 63.2 / 50.2 / 19.7 ms (sum 133.1; r1 99.8) | 35.8 / 13.4 / 10.7 ms (sum 59.9; r1 73.1) | 48.5 / 31.1 / 16.8 ms (sum 96.3; r1 114.3) |
| unattributed (wall − device − split) | 7.8 ms (r1 28.9) | 5.1 ms (r1 47.1) | 1.7 ms (r1 0.0) | 7.4 ms (r1 35.7) |
| PCC normalized | 0.99993 (r1 0.99993) | 0.99980 (r1 0.99977) | 0.99975 (r1 0.99976) | 0.99993 (r1 0.99994) |
| cold (first call) | 0.90 s (r1 4.55 s) | 0.78 s | 0.46 s (r1 13.05 s) | 17.0 s |

- **The forward is still device-bound.**
  - The measured device time (530.9 ms) matches the profiled kernel sum (527.9 ms, −0.6 %). In-section op-to-op gaps are about 0.5 µs.
  - Under Tracy, the host issues the 12-op `time` layer in 1.43 ms (≈ 130 µs per op, start to start), while the device takes 20.19 ms. Host issue is therefore ≈ 43 ms per forward and fully hidden.
  - "(wall − device)/ops" is fixed per-call host work, not dispatch cost.
- **The host split is now a fifth of wall time, and it is noisy.**
  - prepare + upload + postprocess = 141.0 ms (20.7 % of wall), against 105.5 ms in round 1. The host code is unchanged.
  - The R2-2 candidate bench ran the identical code and measured 51.4 / 33.1 / 16.9 = 101.4 ms, with 642.7 ms wall. Each split is taken from a single run after the 5 steady runs, not as a median. A background `/tmp/eval_full.py` process has been using ≈ 100 % of one CPU since Oct 19, which inflates and scatters host times (accuracy wall runs span 634–686 ms; round 1 spanned 833–841).
  - A CPU-only re-timing at the paper shape, with no device opened, gives:
    - `prepare_inputs`: 33–41 ms, of which the float64 causal statistics are 24 ms.
    - `ttnn.from_torch` of the fp32 tokens to bf16: 21.6 ms. If the tensor is cast to bf16 in torch first, the cast takes 0.8 ms and `from_torch` 2.9 ms.
    - `postprocess_head` host math: 11 ms.
- **The device time fell by 171.1 ms, in line with the candidates.** R2-4 −15.6 ms, R2-1 −96.3 ms, R2-3 −34.2 ms, R2-2 −18.5 ms (sum −164.6). The final bench measured 6.5 ms less than the R2-2 candidate for the same code (530.9 vs 537.4 ms), which is ±0.6 % run-to-run noise.
- **Performance precision is now slower on device than accuracy:** 549.9 vs 530.9 ms (+19.0 ms). The fused residual add does not run with bf8 weights; see §8.
- **Multivariate: the blocked variate attention is still untouched.** Multivariate − paper device = **341.0 ms** (round 1: 357.3 ms, round 0: 361.7 ms). That is 56.8 ms extra per `time_variate` layer, 39 % of multivariate device time. It is still not profiled (F11).
- long_context runs at 6.29 µs of device time per token (32 x 1056 tokens; round 1: 8.08). Paper runs at 5.40 µs per padded token (round 1: 7.14), which is 7.86 µs per real token.
- Cold times recovered: paper 4.55 → 0.90 s, long_context 13.05 → 0.46 s. Multivariate cold is 17.0 s, presumably compiling its variate-path kernels (permute, padded SDPA); it was not broken down.

## 2. Reconciliation

Estimate from the reduced profile: **527.86 ms**. Measured device: **530.90 ms** (estimate −0.6 %). The 3 ms difference is within run-to-run noise: the R2-2 candidate measured 537.4 ms for identical code. The profile syncs at every section boundary. tt-perf-report's "tracing could save 1051 µs" comes from those inflated first gaps of each section, which are excluded from device time.

| section | device ms (1 copy) | ops | multiplier | contribution ms | % | round 1 contribution ms |
|---|---|---|---|---|---|---|
| embed | 9.29 | 6 | 1 | 9.29 | 1.8 | 9.30 |
| layer_time | 20.19 | 12 | 18 | 363.43 | 68.9 | 489.79 (27.21 x 18) |
| layer_time_variate | 25.73 | 17 | 6 | 154.40 | 29.3 | 200.35 (33.39 x 6) |
| head | 0.73 | 7 | 1 | 0.73 | 0.1 | 0.74 |
| **total** | | | | **527.86** (331 ops) | 100 | 700.18 (457 ops) |

`layer_time_variate` − `layer_time` = 5.54 ms (round 1: 6.18 ms). That is the univariate variate sublayer (IDs 78–82: norm 724, wg + sigmoid 1359, wv 1079, MUL 999, wo + residual 1268 µs = 5.43 ms), plus 84 µs of SwiGLU variance. Across the 6 layers it is 32.6 ms.

Per `time` layer: matmuls 10.89 ms (53.9 %), everything else 9.30 ms (46.1 %). Round 1 was 43.4 % / 56.6 %.

## 3. Category split and matmul utilization

- Full model (sections x multipliers): **Compute 90.1 % / TM 9.9 % / Other 0 %**. Round 1 had the same split.
  - Compute 475.6 ms = matmuls 290.9 (55.1 %) + rotary 78.1 (14.8 %) + RMSNorm 39.0 (7.4 %) + SDPA 34.6 (6.5 %) + BinaryNg 33.2 (6.3 %).
  - TM 52.2 ms = create heads 51.8 + head slice/reshape 0.5. The concat heads are gone (written by SDPA).
- Reduced window as printed by tt-perf-report: Compute 56.4 / Other 35.1 / TM 8.6. "Other" is `MinimalMatmulDeviceOperation`, which tt-perf-report still does not classify ("Unclassified operation" warning). Reclassified as compute, the reduced window is 91.4 / 8.6 / 0.
- **Weighted mean matmul FLOPs utilization:**
  - Reduced window: tt-perf-report prints 46.79 % (`MinimalMatmulDeviceOperation`, 10 ops) and 32.47 % (2D `MatmulDeviceOperation`, 9 ops; per-op-core basis) separately. Combined: ≈ 41 %.
  - **Full model: 43.9 % (304 basis) / 37.2 % (359 basis).** Round 1: 41.1 % / 34.8 %; round 0: 15.6 % / 13.2 %.
  - The three big matmuls (fused SwiGLU, QKV, w_down + residual): 50.6 % / 42.9 % (round 1's four big ones: 44.9 / 38.0).
  - The 66 768x768 matmuls (wg, wo + residual, wv): 29.4 % / 24.9 % (round 1: 31.5 / 26.7). wo now also reads the residual, so its FLOPs % fell while the ADD it replaced disappeared.
- All matmuls together do 38.8 TFLOP per forward (unchanged: the fused SwiGLU does the same math as up + gate). That is 108.1 ms of math at 359 TFLOPs (127.6 ms at 304), against **290.9 ms** measured (round 1: 310.8 ms).
- The per-op roofline (max of logical DRAM bytes / 512 GB/s and FLOPs / 359 TFLOPs) sums to **154.0 ms** (161.6 ms on the 304 basis). It is up from 149.0 ms because the fused residual read is now counted in the matmuls. The matmuls are **1.89x** off their roofline (round 1: 2.09x); the gap is 136.9 ms.
- Arithmetic intensity:
  - The fused SwiGLU is the only matmul above the HiFi2 ridge of 701 FLOP/B: 1104 FLOP/B, so its roofline is math (1.72 ms at 359 TFLOPs).
  - The others are below the ridge and therefore DRAM-floor-bound: QKV 573 FLOP/B, w_down + residual 437, wo + residual 256, wg/wv 382.
- Whole graph: **153.0 GB of logical DRAM traffic per forward** (matmuls 71.1 GB, everything else 81.9 GB; round 1: 219.0 GB). The DRAM floor is **298.8 ms** (round 1: 427.7 ms); the forward runs at 1.77x that floor (round 1: 1.64x).

**Exit criterion (a), main matmuls > 65 % FLOPs AND non-matmul ops < 15 % of device time: NO on both counts.**

| condition | target | measured |
|---|---|---|
| main matmul FLOPs % (304 / 359 basis) | > 65 % | fused SwiGLU 54.7 / 46.4, QKV 44.2 / 37.4, w_down + residual 50.9 / 43.1 (weighted 50.6 / 42.9); 768x768: wg + sigmoid 28.0 / 23.7, wo + residual 29.8 / 25.2, wv 35.4 / 29.9 |
| non-matmul share of device time | < 15 % | **44.9 %** (237.0 of 527.9 ms; round 1: 55.6 %) |

- Some of the matmuls cannot reach 65 % at their DRAM floor:
  - wo + residual reads three E-sized tensors, so its FLOPs-% ceiling at the DRAM floor is 43 %.
  - wg/wv have a ceiling of 64.4 %.
  - w_down + residual (73.6 %), QKV (96.4 %) and the SwiGLU (compute-bound) can reach it.
- **The non-matmul condition is unreachable with stock ops.** At their DRAM floors, create heads (42.5 ms), RoPE (28.3), SDPA (28.3), RMSNorm (32.4) and the gate MUL (26.5) add up to 158 ms, more than the 154 ms matmul floor. Even F1's fused head-split path leaves ≈ 144 ms of non-matmul floor. Getting under 15 % would need the norms, RoPE, head splitting and the gate fused into the matmul/SDPA kernels.

**Exit criterion (b), round-over-round improvement: this round improved device time by 24.4 % (702.0 → 530.9 ms) and wall by 18.7 % (836.4 → 679.6 ms). This is not a plateau.**
- Headroom remaining: the DRAM floor of the current graph is 298.8 ms (1.77x away), and the matmul roofline gap is 136.9 ms.
- The findings below add up to a realistic **60–90 ms of device time (−11 to −17 %)** for the next round: F1 or F2, F5, F6, F9 and F10. On top of that come **30–45 ms of host wall** (F3) and ≈ **33 ms in performance precision only** (F4).

## 4. Per-op-type table (full model)

| op type | full-model ms | % | count | round 1 ms | round 1 count | change ms |
|---|---|---|---|---|---|---|
| MinimalMatmulDeviceOperation (incl. fused SwiGLU and addcmul) | 184.74 | 35.0 | 86 | 82.82 | 62 | +101.92 |
| MatmulDeviceOperation (2D multicast) | 106.15 | 20.1 | 58 | 228.00 | 106 | −121.85 |
| *all matmuls* | *290.89* | *55.1* | *144* | *310.82* | *168* | ***−19.93*** |
| RotaryEmbeddingLlamaDeviceOperation (r1: RotaryEmbedding) | 78.06 | 14.8 | 48 | 95.84 | 48 | **−17.78** |
| NlpCreateHeadsDeviceOperation | 51.77 | 9.8 | 24 | 51.65 | 24 | +0.12 |
| LayerNormDeviceOperation (RMSNorm, HiFi4) | 38.95 | 7.4 | 55 | 39.13 | 55 | −0.18 |
| SDPAOperation (now with `output_concat_heads`) | 34.56 | 6.5 | 24 | 32.75 | 24 | +1.81 |
| BinaryNg MUL | 31.14 | 5.9 | 31 | 96.20 | 55 | **−65.06** |
| BinaryNg ADD | 2.02 | 0.4 | 3 | 56.04 | 57 | **−54.02** |
| NLPConcatHeadsDeviceOperation | 0 | 0 | 0 | 17.28 | 24 | **−17.28** |
| Slice + ReshapeView (head) | 0.47 | 0.1 | 2 | 0.47 | 2 | 0 |
| **total** | **527.86** | 100 | **331** | 700.18 | 457 | **−172.32** |

- Grouped by base op: BinaryNg 33.16 ms (6.3 %, 34 ops: 24 ctx x gate, 6 v x gate, 3 embed, 1 head); all non-matmul ops 236.97 ms (44.9 %, 187 ops).
- What is left of BinaryNg is the attention gate MUL (30 x 1.0 ms) and the embed/head elementwise ops.

Matmuls by role (round 1 figures include the ops each fusion absorbed):

| role | n | round 1 ms | round 2 ms | per call µs | round 2 engine | change ms |
|---|---|---|---|---|---|---|
| MLP front half: up + SILU, gate, MUL → fused SwiGLU | 24 | 186.6 (66.9 + 54.8 + 64.9) | 88.70 | 3717 | minimal `fuse_swiglu`, M/K/N 8/12/2, subblock 4x2 | **−97.9** |
| QKV | 24 | 62.2 | 62.17 | 2592 | 2D 11x10 | 0 |
| MLP down + residual ADD | 24 | 64.4 (40.4 + 24.0) | 47.98 | 1999 | addcmul (minimal 8/8/8) | −16.4 |
| attn gate + SIGMOID, 24 time + 6 variate | 30 | 40.8 | 40.76 | 1360 | 2D 8x10 (80 cores), SIGMOID fused | 0 |
| attn wo + residual ADD, 24 + 6 | 30 | 62.5 (32.5 + 30.0) | 38.31 | 1277 | addcmul (minimal 8/8/8) | −24.2 |
| variate wv | 6 | 6.6 | 6.47 | 1079 | minimal 8/8/8 | −0.1 |
| embed (3 matmuls) | 3 | 6.29 | 6.28 | — | as round 1 | 0 |
| head (3 matmuls + SILU) | 3 | 0.22 | 0.21 | — | 2D | 0 |
| **total (incl. absorbed ops)** | **144** | **429.6** | **290.9** | | | **−138.7** |

Two costs of the fusions:
- The fused residual adds +193 µs to wo (1084 → 1277 µs) and +315 µs to w_down (1684 → 1999 µs), in exchange for a 1000 µs ADD each.
- The fused SwiGLU (3717 µs) replaces 7.78 ms of up + gate + MUL.

## 5. Op-to-op map

All ops run on 110 cores (except the 80-core gate matmuls and the 64-core head norm), DRAM interleaved, bf16, HiFi2 for matmuls. RoPE and RMSNorm run at HiFi4. Configs come from `tt/common.py` → `tt/program_configs.py`:
- "2D" = `MatmulMultiCoreReuseMultiCastProgramConfig` with `in0_block_w=12`, `out_block_h=11`, `per_core_M=308` (= 28 x 11), `out_subblock_h=1`.
- "minimal" = `ttnn.experimental.minimal_matmul` with M/K/N blocks 8/8/8, subblock 1x8, grid 11x10.
- "addcmul" = `ttnn.experimental.dit_minimal_matmul_addcmul_fused(x, W, 1.0, residual, ones[1, N])` on the minimal config.
- "SwiGLU" = `minimal_matmul(fuse_swiglu=True)` on the tile-pair-interleaved [silu | mul] weight. Blocks M/K/N = 8/12/2, subblock 4x2. The core grid is transposed (M > N): M is spread over 11 cores (280 M tiles each) and the 64 output tiles over 10 cores (7 per core, so 70 padded, 9.7 % idle work).

Bound: tt-perf-report for matmuls; `DRAM*`/`SLOW*` is my classification (estimated DRAM % at least 65 % or not).

### One `time` layer (IDs 55–66, 20.19 ms; round 1 27.21 ms; round 0 50.24 ms)

| ID | op | shape / M x K x N | µs | bound | DRAM % | FLOPs % (304 / 359) | config | source |
|---|---|---|---|---|---|---|---|---|
| 55 | rms_norm (attn) | 98304x768 | 717 | DRAM* | 82* | — | default, HiFi4 | attention.py:88 → common.py:222 |
| 56 | linear wg + SIGMOID | 98304x768x768 | 1357 | SLOW | 43.6 | 28.1 / 23.8 | 2D 8x10, per_core_N=3, sb 1x3 | attention.py:89 → common.py:140 |
| 57 | linear wqkv | 98304x768x2304 | 2589 | SLOW | 45.8 | 44.2 / 37.4 | 2D 11x10, per_core_N=7, sb 1x7 | attention.py:170 |
| 58 | split_qkv_and_split_heads | 98304x2304 → 3 x (1024,12,96,64) | 2157 | DRAM* | 82* | — | | attention.py:110 (via :171) |
| 59 | rotary_embedding_llama q | (1024,12,96,64) | 1628 | DRAM** | 36* logical, ≈ 72 on actual traffic | — | reload path (36 rows/core > 8), HiFi4 | attention.py:151 (via :172) |
| 60 | rotary_embedding_llama k | (1024,12,96,64) | 1627 | DRAM** | 36* / ≈ 72 | — | same | attention.py:151 (via :172) |
| 61 | SDPA causal, `output_concat_heads` | q/k/v (1024,12,96,64) → (1024,1,96,768) | 1442 | DRAM* | 82* | 3.3 (PM) | chunk 96 | attention.py:126 (via :173) |
| 62 | MUL ctx * gate | 98304x768 | 1005 | DRAM* | 88* | — | SFPU mul | attention.py:97 |
| 63 | wo + residual (addcmul) | 98304x768x768 + residual | 1272 | DRAM* | 69.7* (tpr 46.5) | 30.0 / 25.4 | addcmul | attention.py:103 → common.py:179 |
| 64 | rms_norm (MLP) | 98304x768 | 723 | DRAM* | 82* | — | default, HiFi4 | mlp.py:43 |
| 65 | fused SwiGLU | 98304x768x(2x2048) → 98304x2048 | 3675 | SLOW | 29.8 | 55.3 / 46.9 | SwiGLU 8/12/2, sb 4x2 | mlp.py:44 → common.py:208 |
| 66 | w_down + residual (addcmul) | 98304x2048x768 + residual | 1999 | DRAM* | 69.1* (tpr 54.4) | 50.9 / 43.1 | addcmul | mlp.py:49 → common.py:179 |

`**` RoPE: the logical bytes (read q, write q) are 302 MB, which is 36 % of peak. The reload path also reads the 2 sin + 2 cos tiles for every (batch, head, seq-tile) row of 2 input tiles. That makes the actual traffic 604 MB, ≈ 72 % of peak, so the op is DRAM-bound on traffic it does not need (F2).

### One `time_variate` layer (IDs 69–85, 25.73 ms; round 1 33.39 ms; round 0 63.77 ms)

| ID | op | shape / M x K x N | µs | bound | DRAM % | FLOPs % (304 / 359) | source |
|---|---|---|---|---|---|---|---|
| 69 | rms_norm (time attn) | 98304x768 | 724 | DRAM* | 82* | — | attention.py:88 |
| 70 | linear wg + SIGMOID (time) | 98304x768x768 | 1363 | SLOW | 43.4 | 28.0 / 23.7 | attention.py:89 |
| 71 | linear wqkv | 98304x768x2304 | 2595 | SLOW | 45.7 | 44.1 / 37.3 | attention.py:170 |
| 72 | split_qkv_and_split_heads | → 3 x (1024,12,96,64) | 2158 | DRAM* | 82* | — | attention.py:110 |
| 73 | rotary_embedding_llama q | (1024,12,96,64) | 1623 | DRAM** | 36* / ≈ 72 | — | attention.py:151 |
| 74 | rotary_embedding_llama k | (1024,12,96,64) | 1622 | DRAM** | 36* / ≈ 72 | — | attention.py:151 |
| 75 | SDPA causal, concat-heads out | (1024,12,96,64) → (1024,1,96,768) | 1435 | DRAM* | 82* | 3.3 (PM) | attention.py:126 |
| 76 | MUL ctx * gate | 98304x768 | 1007 | DRAM* | 88* | — | attention.py:97 |
| 77 | wo (time) + residual | 98304x768x768 + residual | 1301 | DRAM* | 68.2* | 29.3 / 24.8 | attention.py:103 |
| 78 | rms_norm (variate attn) | 98304x768 | 724 | DRAM* | 81* | — | attention.py:88 |
| 79 | linear wg + SIGMOID (variate) | 98304x768x768 | 1359 | SLOW | 43.6 | 28.1 / 23.8 | attention.py:89 |
| 80 | linear wv (univariate V-path) | 98304x768x768 | 1079 | SLOW | 54.9 | 35.4 / 29.9 | attention.py:93 |
| 81 | MUL v * gate | 98304x768 | 999 | DRAM* | 89* | — | attention.py:97 |
| 82 | wo (variate) + residual | 98304x768x768 + residual | 1268 | DRAM* | 70.0* | 30.1 / 25.5 | attention.py:103 |
| 83 | rms_norm (MLP) | 98304x768 | 720 | DRAM* | 82* | — | mlp.py:43 |
| 84 | fused SwiGLU | 98304x768x(2x2048) | 3759 | SLOW | 29.1 | 54.1 / 45.8 | mlp.py:44 |
| 85 | w_down + residual | 98304x2048x768 + residual | 2000 | DRAM* | 69.1* | 50.9 / 43.1 | mlp.py:49 |

Configs are as in the `time` layer: 2D for wg/wqkv, minimal for wv, addcmul for wo/w_down, SwiGLU for the MLP. This is the last layer of the model (block schedule (time, time, time, time_variate) x 6). Only its decoder rows reach the head (F5).

Embed (IDs 47–52, 9.29 ms; round 1 9.30 ms), unchanged:
- 47: hidden 98304x64x3072 + bias + SILU, 3007 µs. 2D, `in0_block_w=2`; DRAM 40.1 %, FLOPs 4.2 % (residual_block.py:43).
- 48: output 98304x3072x768 + bias, 2625 µs. Minimal; DRAM 56.5 %, FLOPs 58.1 % (residual_block.py:44).
- 49: residual 98304x64x768 + bias, 648 µs. Minimal (residual_block.py:46).
- 50: ADD main + skip, 994 µs.
- 51: MUL keep, broadcast, 1015 µs (model.py:198).
- 52: ADD query, 999 µs (model.py:200).

Head (IDs 88–94, 0.73 ms): slice 268 µs (model.py:211), reshape copy 206 µs (:213), rms_norm 21 µs on 64 cores (:214), 2D matmuls hidden + SILU 103 µs, output 83 µs, residual 30 µs, ADD 24 µs.

## 6. Top 10 op roles (full model, with multiplicity)

The lower bound (LB) per call is max(FLOPs / 359 TFLOPs, logical bytes / 512 GB/s), where bytes = inputs (incl. the fused residual) + weight + output in bf16. Headroom = (actual − LB) x n.

| # | role | n | full ms | per call µs | M x K x N / shape | cores | DRAM % | FLOPs % (304 / 359) | limit hit | LB/call µs | headroom ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fused SwiGLU 768 → 2x2048 | 24 | 88.7 | 3717 | 98304x768x4096 → 2048 | 110 | 29.4 | 54.7 / 46.4 | neither: not FPU-bound (LoFi gains ≤ 0.4 ms), 9.7 % padded N, in0 re-read 4x, SFPU epilogue | 1723 (math; DRAM 1094) | **47.9** |
| 2 | rotary q + k (llama, reload) | 48 | 78.1 | 1625 | (1024,12,96,64) | 110 | 36.3* (≈ 72 actual) | — | DRAM on actual traffic: cos/sin re-read per (batch, head, row) | 590 (DRAM) | **49.7** |
| 3 | QKV 768x2304 (2D) | 24 | 62.2 | 2592 | 98304x768x2304 | 110 | 45.8 | 44.2 / 37.4 | neither: 2D pipeline, per_core_N 7 (7 % padded N) | 1187 (DRAM; math 969) | **33.7** |
| 4 | create heads (split qkv) | 24 | 51.8 | 2157 | 98304x2304 → 3 x (1024,12,96,64) | 110 | 82.0* | — | DRAM | 1769 | 9.3 |
| 5 | w_down 2048x768 + residual | 24 | 48.0 | 1999 | 98304x2048x768 + 98304x768 | 110 | 69.1* | 50.9 / 43.1 | DRAM (near) | 1382 (DRAM) | 14.8 |
| 6 | attn gate 768x768 + SIGMOID (2D), 24 time + 6 variate | 30 | 40.8 | 1360 | 98304x768x768 | **80** | 43.5 | 28.0 / 23.7 | neither: 80 cores + in-kernel sigmoid | 592 (DRAM; math 323) | **23.0** |
| 7 | rms_norm | 55 | 39.0 | 721 | 98304x768 | 110 | 81.7* | — | DRAM | 590 | 7.1 |
| 8 | wo 768x768 + residual, 24 + 6 | 30 | 38.3 | 1277 | 98304x768x768 + 98304x768 | 110 | 69.1* | 29.8 / 25.2 | DRAM (near) | 887 (DRAM) | 11.9 |
| 9 | SDPA causal + concat-heads output | 24 | 34.6 | 1438 | q/k/v (1024,12,96,64) | 110 | 82.0* | 3.3 (PM) | DRAM | 1180 | 6.2 |
| 10 | attn gate MUL (ctx or v) * gate | 30 | 30.1 | 1004 | 98304x768 | 110 | 88.1* | — | DRAM | 885 | 3.6 |

- Together these ten roles are 511.4 ms (96.9 %).
- Matmul gaps per call: wg 2.30x LB, QKV 2.18x, SwiGLU 2.16x, w_down + residual 1.45x, wo + residual 1.44x.
- Next in line: variate wv (6 x 1079 µs = 6.5 ms, 1.82x LB, 2.9 ms headroom), embed (9.3 ms, 6 ops, ≈ 4.2 ms headroom), head (0.7 ms).

Notes on the roles the brief singles out:
- **Fused SwiGLU** (#1).
  - What limits it: the minimal_matmul loop is m-block → n-block → k-block, and only the last K block is reused when stepping N (`dm_in0_sender.cpp:251-290`, `:356-359`, `:480-481`). With K = 24 tiles in two K blocks of 12 and 7 N blocks per core, every activation block is read 8 times instead of 2, so **in0 traffic is 4x**: ≈ 604 MB from DRAM.
  - The 6.3 MB weight slab is re-streamed for each of the 35 M blocks per core, ≈ 220 MB.
  - Actual traffic is therefore ≈ 1.23 GB per call, ≈ 65 % of DRAM peak at 3.7 ms. Fewer re-reads did not help, though:
    - `[8,24,2]` (K in one block, in0 read once) is slower at 4.56 ms.
    - `[8,8,14]` (one N block, in0 read once) takes 4.44 ms.
  - LoFi barely helps (§8). So neither raw DRAM nor the FPU is the single limiter; the per-block pipeline, including the serialized SFPU silu + mul epilogue on TRISC1, is. A device-zone profile of this op (tt-profiler skill) is needed to split it further.
  - The best sweep point, `[16,12,2]`, is only 1 % faster (3.806 vs 3.847 ms; ≈ 1 ms full model). `[16,12,6]` fails CB allocation.
- **wqkv** (#3) is untouched since round 1. If F1 moves it to `minimal_matmul_split`, it needs a re-sweep.
- **wg + sigmoid** (#6) still runs on 80 cores, because N = 24 tiles does not split over 11 or 10 columns. R2-5 (sigmoid in the MUL, wg on minimal) measured −3.9 ms and was rejected as too small. Packing it into the QKV matmul (F6) is worth 12–18 ms.
- **Rotary** (#2) is now 1625 µs (from 1997). The non-reload path does not help by itself: the `(R·H, 1, S, Dh)` layout takes it (3 rows/core) but still measured 1.678 ms, because it re-reads the batch-broadcast cos/sin once per batch (F2).
- **Create heads** (#4) and **RMSNorm** (#7) are at 82 % of DRAM peak. They can only go away through fusion (F1, F10).
- **Gate MUL** (#10) is at 88 % DRAM; only fusion removes it (F9).

## 7. Findings, ranked by estimated full-model savings

Ranking is by estimated savings weighted by confidence. Savings overlap and are not additive (F1 and F2 are alternatives). All estimates are full-model device ms at the paper shape, accuracy precision, unless marked wall or performance-only.

**F1. Fused head split + RoPE: `minimal_matmul_split` → `wan_fused_rmsnorm_post_allgather` (identity stats, RoPE fused, per-head output) → SDPA on an (H, R, S, Dh) view → `nlp_concat_heads`. Savings ≈ 45–55 ms (+24 ops). Confidence: low-medium. Model-level only, existing ops.**

- Evidence rows: 57–61 and 71–75. Today the time-attention core is create heads 2157 + RoPE 2 x 1625 + SDPA/concat 1438 = **6.85 ms per layer, 164.4 ms over 24 layers (31 % of device time)**. Create heads and RoPE alone move ≈ 1.8 GB per layer.
- Mechanism:
  1. `ttnn.experimental.minimal_matmul_split(xn, Wqkv, chunks=3)` writes q, k, v as three flat [1, 1, 98304, 768] tensors. N = 72 tiles is divisible by 3, and the weight dtype matches the input (`minimal_matmul_split_nanobind.cpp`).
  2. `wan_fused_rmsnorm_post_allgather(t, stats, num_heads_per_device=12, transformation_mat, rope_cos, rope_sin)` writes [1, 12, 98304, 64] with RoPE applied (`rmsnorm_distributed_nanobind.cpp:122-172`). It takes batch-1 [1, 1, S, H] input, cos/sin [1, 1, S, 64] in the same pair order used today, and the existing (1, 1, 32, 32) trans_mat.
     - RoPE is optional, so V uses the same op without it.
     - Stats: a constant tensor built once, e.g. `wan_fused_rmsnorm_pre_allgather(ones)` with eps = 0, makes the norm scale exactly 1.
  3. [1, 12, 98304, 64] is a free view of (12, 1024, 96, 64): batch = head, heads = rows. SDPA is per-(batch, head) independent, so causal SDPA on that view is exact, with 12288 work units as today.
  4. The SDPA output viewed as [1, 12, 98304, 64] → `nlp_concat_heads` → [1, 1, 98304, 768]. `output_concat_heads` cannot be used here, because it would concatenate the 1024 "heads".
- Expected per layer: 3 post ops at 0.75–1.0 ms each (302 MB each, like create heads at 82 %), plus SDPA 1.37 ms (round 1, without the concat output) and concat heads 0.72 ms, = 4.3–4.8 ms against 6.85 ms. That is **−2.1 to −2.6 ms per layer**. wqkv is assumed unchanged (the round 1 sweep had minimal 2.25 vs 2D 2.28 ms).
- Risks:
  - The post op is built for the wan DiT (S ≈ 10–75k, batch 1). Its speed at S = 98304 and its numerics with identity stats need a micro-benchmark first.
  - It needs full-length cos/sin tables: 12.6 MB each, +25 MB of reads per call.
- It subsumes F2. It also enables F6 (wg as a 4th split chunk) and F10's attention-norm fold.

**F2. RoPE kernel: keep batch-broadcast cos/sin resident across the batch loop. Savings ≈ 35–42 ms. Confidence: medium. A small ttnn kernel change; it is the alternative to F1.**

- Evidence rows: 59, 60, 73, 74. 48 calls of 1625 µs = 78.1 ms (14.8 %).
- Mechanism:
  - Both kernel paths re-read cos/sin. In the reload path (`use_reload_impl = num_rows_per_core > 8`, program factory ≈ line 110), the reader reads 2 input + 2 sin + 2 cos tiles per (batch, head, seq-tile) with a barrier each time.
  - In the non-reload path, the reader re-reserves and re-reads cos/sin for every `batch_id` (`reader_rotary_embedding_llama_interleaved_start_id.cpp:71-83`). The compute kernel also waits and pops them per batch (`rotary_embedding_llama.cpp:98-102`, `:195-198`).
  - Either way the traffic is 604 MB per call instead of 302 MB. That is why the `(R·H, 1, S, Dh)` microbenchmark (non-reload) took 1.678 ms, the same as the reload path's 1.686 ms.
- Fix: when cos/sin have batch 1, hoist the cos/sin reserve/read/push in the reader and the wait/pop in compute out of the batch loop. Per core that is 3 rows x 2 x 2 tiles = 24 KB, held for the core's whole batch range. Use the `(R·H, 1, S, Dh)` view, which already takes the non-reload path, or relax the heuristic to `num_sin_cos_rows_per_core > 8`.
- Expected 0.75–0.9 ms per call (302 MB at 65–80 % of peak): **−0.75 to −0.88 ms x 48**.
- The old 4-read `rotary_embedding` (2.08 ms) and `rotary_embedding_hf` (same reader pattern) are no alternatives. Sharded cos/sin fall back to reload when batch > 1. LoFi gives the same time with lower PCC.

**F3. Host wall: cast to bf16 in torch before `from_torch`, compute the causal stats at patch granularity, and measure the split as medians. Savings ≈ 30–45 ms of wall (no device effect). Confidence: medium-high for the upload, medium for prepare.**

- Evidence: the split is 141.0 ms (20.7 % of wall); a CPU-only re-timing gives prepare 33–41 ms, upload 33–49 ms and postprocess ≈ 11 ms of host math (§1).
- Mechanism:
  - **Upload:** `upload_inputs` sends fp32 tokens with `dtype=bfloat16` (model.py:143), so `ttnn.from_torch` converts on the host in 21.6 ms. `tokens.to(torch.bfloat16)` first costs 0.8 ms, and the bf16 `from_torch` 2.9 ms: **−18 to −19 ms**. Bit-identical.
  - **Prepare:** the float64 cumsum statistics over 2048 steps per row take 24 ms. The head only needs statistics at patch boundaries, so per-patch sums of x and x² (64 per row) followed by a cumsum over patches replace the 2048-step float64 cumsum with a reduction plus a 64-step cumsum: **−15 to −25 ms**. The results are not bit-identical in float64, but the differences are far below bf16.
  - **Postprocess:** `monotone_quantiles` + denormalize over 1.4 M elements in 11 ms can be vectorized for a few ms.
- Measurement: the bench takes the split from one run after the steady runs. Make it a median of the steady runs. Do not benchmark while `/tmp/eval_full.py` is running.
- Trace capture does not help here: the unattributed wall residual is only 7.8 ms.

**F4 (performance precision only). Keep wo and w_down in bf16 in performance mode so the fused residual add runs. Savings ≈ 33 ms of performance-precision device time (549.9 → ≈ 516 ms). Confidence: high.** See §8.

- Evidence: R2-3 saved 34.2 ms in accuracy and 0.7 ms in performance. `linear_residual` takes the addcmul path only if `residual.dtype == weight.dtype` (common.py:178). That guard mirrors the kernel's `TT_FATAL(ternary_a_data_format == in1_data_format)` (`minimal_matmul_program_descriptor.cpp:458`).
- Mechanism: wo (1.2 MB) and w_down (3.1 MB) weights per layer are 0.1 GB of DRAM in total. These ops are DRAM-bound on activations, so bf8 weights buy nothing; LoFi still applies with bf16 weights.
- Alternative: relax the TT_FATAL in ttnn. The compute kernel already reconfigures the unpacker per epilogue step (`compute.cpp:222` for ternary_b); the ternary_a add would need the same. This is medium confidence and gains nothing over the model-side fix.

**F5. Prune the last `time_variate` layer to the decoder rows. Savings ≈ 12–16 ms. Confidence: medium-high. Low complexity.**

- Evidence rows: 70 and 76–85, plus head rows 88–89. The head reads only tokens `ll .. ll+h` of each row (model.py:211), and the last layer is `time_variate`. Nothing after its time-attention SDPA mixes tokens across time slots:
  - The univariate V-path and the MLP are per-token.
  - Multivariate variate attention mixes variates within the same slot, so it needs only the decoder slots of all variates.
- Mechanism: after SDPA (ID 75), slice ctx, the residual x and xn to the decoder tokens, using the head's slice + reshape (≈ 0.47 ms per tensor, IDs 88–89). Then run MUL, wo, the variate sublayer, the norms, wg and the MLP on 2048 rows (≈ 0.03–0.1 ms each), and drop the head's own slice/reshape.
  - Removes 14.2 ms (IDs 76–85) + 1.36 ms (ID 70: compute wg from sliced xn) + 0.47 ms (head).
  - Adds ≈ 1.4 ms of slices and ≈ 0.4 ms of small ops.
  - Net **≈ −14 ms**.
- Extension: also slice q to the decoder rows before RoPE in that layer, for ≈ 2 ms more.
- Long context gains proportionally less (h = 32 of 1056 tokens), but still gains.

**F6. Pack wg into the QKV split matmul (and wg + wv for the V-path), and apply the sigmoid in the gate MUL. Savings ≈ 12–18 ms (−30 ops). Confidence: low-medium. Requires F1's `minimal_matmul_split`.**

- Evidence rows: 56, 70, 79, 80. wg + sigmoid on 80 cores costs 30 x 1360 µs = 40.8 ms at 28 % FLOPs (2.30x LB).
- Mechanism:
  - `minimal_matmul_split(xn, [Wq | Wk | Wv | Wg], chunks=4)` (N = 96 tiles, 24 per chunk) gives q, k, v and g_raw with no slice. Round 1 dropped P5 because it needed a slice and `split_query_key_value_and_split_heads` needs exactly 3·D columns; neither applies to the F1 path.
  - Then `ttnn.mul(ctx, g_raw, input_tensor_b_activations=[SIGMOID])`. R2-5 showed the MUL absorbs the sigmoid at no extra cost.
  - The extra 768 columns cost ≈ 0.75–0.9 ms inside a 110-core matmul at QKV efficiency, against 1.36 ms standalone: **−0.45 to −0.6 ms x 24**.
  - The univariate V-path packs [wg | wv] with `chunks=2`, ≈ −0.5 ms x 6.
- Without F1, the R2-5 variant alone gives −3.9 ms (measured).

**F7. Remove the per-row tile padding (66 of 96 tokens) outside the attention core (R2-7). Savings ≈ 15–40 ms net. Confidence: low. Complexity: high.**

- Evidence: the flat (per-token) ops outside create heads, RoPE and SDPA now total 363.5 ms. 31.25 % of that is pad work: **113.6 ms gross** (round 1: 157 ms).
- Cheapest layout: keep the residual stream unpadded at [67584, 768] (2112 tiles, tile-aligned). Pack xn to the padded layout before wg/wqkv, and unpack the gated ctx after the attention MUL, so wo, RMSNorm, SwiGLU and w_down run unpadded.
  - Per layer that saves ≈ 2.6 ms of flat work.
  - It costs one 768-wide pack and one unpack: row-granular tiled re-layouts of ≈ 255 MB each, ≥ 0.5 ms each at peak and realistically 1 ms.
  - Net ≈ 0.6–1.6 ms x 24.
- Gate: micro-benchmark the pack/unpack pair at the real shape, and proceed only if it costs ≤ 1.2 ms.
- A structural alternative appears in the re-evaluation table (context/decoder split stream).

**F8. Remaining matmul gaps (SwiGLU, QKV, the addcmul pair). Savings ≈ 10–20 ms. Confidence: low.**

- SwiGLU (headroom 47.9 ms):
  - `[16,12,2]` gives −1 ms.
  - `math_approx_mode=True` for the SFPU silu (op-wide, so test the PCC) might save 0.2–0.4 ms per call (5–10 ms).
  - bf8 hidden output (performance precision only; it halves the 403 MB write and w_down's in0 read) gives ≈ 0.3–0.5 ms per layer.
  - Anything beyond that needs the per-zone profile described in §6.
- QKV (headroom 33.7 ms): re-sweep it as a split matmul if F1 lands. Otherwise the round 1 sweep is flat (2D 2.277 vs minimal 2.251 ms).
- wo / w_down + residual: already at 69 % of DRAM peak, so ≤ 5–10 ms remains at a realistic 85 % ceiling.

**F9. Fuse the attention gate MUL. Savings ≈ 3 ms model-only, ≈ 15–20 ms with a kernel change. Confidence: low.**

- Evidence rows: 62, 76, 81. 30 x 1004 µs = 30.1 ms at 88 % DRAM.
- Model-only, univariate V-path: `dit_minimal_matmul_addcmul_fused(xn, Wv, 1.0, zeros[M,768], gate[M,768])` computes (xn @ Wv) ⊙ gate. It supports full [M, N] ternary_b streaming. This needs a resident 151 MB zeros tensor and adds a 151 MB gate read: ≈ −0.5 ms x 6.
- For the time path, the gate multiplies the matmul input (ctx), so no output-side epilogue can absorb it. Options:
  - Multiply in the SDPA output stage: the `output_concat_heads` tiles are the same [M, 768] tiles as the gate.
  - With F1, multiply in `nlp_concat_heads`.
  - An in0 prologue in minimal_matmul.
- Each of those removes the MUL for ≈ +0.3 ms of gate reads in the host op, ≈ −0.7 ms x 24. All are kernel changes.

**F10. RMSNorm configs and small cleanups. Savings ≈ 5–10 ms. Confidence: low-medium.**

- RMSNorm: 55 x 721 µs = 39.0 ms at 82 % DRAM. A program config (block/sharded variant) could take it to ≈ 90 %, saving ≤ 5–7 ms.
- RMSNorm fusion is limited:
  - The MLP norm cannot fold into the SwiGLU, because silu(r·u) ≠ r·silu(u).
  - With F1, the attention norm's row scale could move into the post-op stats for q/k/v (RoPE and V are linear per token). The gate still needs xn before the sigmoid, so the norm stays unless the gate gets its own scaled path. Net ≈ 0.
- Embed: fold main + skip into the embed out matmul via addcmul (≈ 0.6 ms), and replace MUL keep + ADD query with one `ttnn.addcmul` (≈ 1.0 ms).
- Head: folded into F5.

**F11 (multivariate only). Variate attention in row-major order with a block-diagonal mask. Savings ≈ 200–300 ms on multivariate (−25 to −35 %). Confidence: low. Profile first.**

- Evidence: multivariate − paper = 341.0 ms of device time, 56.8 ms per `time_variate` layer, unprofiled.
- `_variate_context` (attention.py:176-193) does the following per layer:
  - permutes the 453 MB QKV to slot-major;
  - reshapes it to (S·B, V=8, 3D), a 4x tile-padding inflation;
  - splits heads and runs SDPA on (12288, 12, 32, 64);
  - permutes back.
- Mechanism: rows are ordered r = b·V + v, so (R, S, 3D) is already a free view of (B=128, V·S = 768, 2304).
  - Create heads as in the time path gives (128, 12, 768, 64), with no permute and no padding inflation.
  - Run SDPA non-causally with an additive mask (1, 1, 768, 768) that lets a token see only same-slot tokens (∧ the variate mask). That is exactly the same attention.
  - Cost: create heads ≈ 2.2 ms, plus SDPA ≈ 4–10 ms for 232 GFLOP and 0.9 G masked exps, plus mask reads. Against 56.8 ms + wv today, that saves ≈ 40–50 ms per layer.
- Prerequisite: a reduced multivariate profile, to confirm where the 56.8 ms goes and to check that a dense 768x768 mask SDPA is as fast as estimated.

### Re-evaluation of the deferred and rejected proposals

| proposal | round 1 estimate | worth now | why |
|---|---|---|---|
| R2-5 sigmoid into MUL, wg on minimal (rejected) | 8 ms | **12–18 ms only together with F1** (F6) | Measured −3.9 ms alone. Packed into the split QKV matmul, it also removes the 80-core matmul. |
| R2-6 host prepare/upload/postprocess | 30–60 ms wall | **30–45 ms wall** (F3) | Now timed CPU-only: 19 ms is a dtype-conversion fix, 15–25 ms is the float64 stats. The bench split itself is noisy (single run, loaded host). |
| R2-7 per-row padding outside attention | 40–90 ms (0–50 after fusions) | **15–40 ms net** (F7) | The fusions shrank the flat work to 363.5 ms (gross 113.6 ms), but the pack/unpack cost does not shrink. Gate on a micro-benchmark. |
| Multivariate variate attention | ≥ 200 ms on multivariate | **200–300 ms on multivariate** (F11) | Unchanged at +341 ms. The block-diagonal-mask layout avoids both permutes and the 4x padding. Profile first. |
| Pack wg into wqkv (round 1 P5, dropped) | ≈ 0 | **12–18 ms** (F6) | `minimal_matmul_split` removes the slice that made it a wash. |
| Precision per tensor group | LoFi ≈ 35 ms after SwiGLU fusion | **performance only:** bf16 wo/w_down ≈ 33 ms (F4); bf8 MLP hidden 7–12 ms; bf8 residual stream more, but risky | LoFi gains on the fused SwiGLU turned out small (≤ 0.4 ms/call). RoPE must stay bf16/HiFi4. Nothing applies to accuracy precision. |
| L1-resident M-chunks | 50–100 ms after fusions | **25–50 ms; very low confidence; later** | The remaining big DRAM round trip is the SwiGLU → w_down hidden (806 MB per layer). The rest is attention-core traffic, which F1/F2 attack directly. |
| Context/decoder split stream (new) | — | **60–110 ms; very low confidence; design study only** | The 30 pad tokens exist only because 2 decoder tokens push 64 context tokens over a tile boundary. Context tokens never attend to decoder tokens (causal). A tile-aligned context stream (S = 64; attention core −33 %, flat ops −31 %) plus a small decoder stream that attends to the context K/V avoids all row-granular re-layouts. The decoder-side K/V concat and RoPE costs are uncertain (≈ 2–3 ms per layer). |
| Trace capture / dispatch | no device gain | **no device gain; ≤ 8 ms wall** | Host issue (≈ 43 ms per forward under Tracy) is hidden behind 531 ms of device time. The program cache hits in steady state. The unattributed wall residual is 7.8 ms. |

Other ideas I checked and rejected:
- Flipping the RoPE reload heuristic alone: the non-reload path also re-reads cos/sin per batch (1.678 ms measured, F2).
- RoPE at LoFi: same time, PCC 0.99990 vs 0.99999.
- SwiGLU with K in one block (`[8,24,*]`, in0 read once): 3.99–4.56 ms, and PCC 0.99974.
- SDPA `output_concat_heads` on the F1 (H, R, S, Dh) view: it concatenates the wrong axis.

**Structural floor.** The current graph moves 153.0 GB per forward, a 298.8 ms floor. With F1 (or F2), F5, F6, F9 (model part) and F10 at their midpoints, device time would be ≈ 531 − 50 − 14 − 15 − 3 − 5 ≈ **445 ms** (≈ 330 ops: F1 adds 24, F6 removes 30). F9's model-only V-path variant conflicts with F6's [wg | wv] packing; pick one. With F2 instead of F1 (so without F6), it would be ≈ 470 ms. F3 takes a further 30–45 ms off wall, and F4 brings performance precision ≈ 15 ms under accuracy.

## 8. Comparison with the performance precision (bf8 weights, LoFi)

- Device: 530.9 → 549.9 ms (**+19.0 ms, +3.6 %, slower**); round 1 was −31.8 ms. Wall: 679.6 → 688.0 ms (+8.4 ms). The wall runs are noisy (performance 647–726, accuracy 634–686), and the split differs by 7.9 ms with identical host code.
- History of the gap, cumulative on the paper shape:

| step | accuracy device | performance device | performance − accuracy |
|---|---|---|---|
| round 1 final | 702.0 | 670.2 | −31.8 |
| + R2-4 SDPA concat-heads | 686.4 | 654.3 | −32.1 |
| + R2-1 fused SwiGLU | 590.1 | 568.5 | **−21.6** |
| + R2-3 residual inside wo / w_down | 555.9 | 567.8 | **+11.9** |
| + R2-2 `rotary_embedding_llama` | 537.4 | 549.8 | +12.4 |
| final bench (same code as R2-2) | 530.9 | 549.9 | +19.0 |

- **Cause 1: the fused residual add does not run with bf8 weights (+33.5 ms relative).**
  - In performance mode the residual is bf16 and the weights bf8, so `linear_residual` (common.py:178) falls back to minimal_matmul + `ttnn.add`. That is 54 extra ADDs (≈ 54 ms), partly offset by the cheaper plain matmuls (≈ −13.5 ms). R2-3 gave −34.2 ms in accuracy and −0.7 ms in performance.
  - The guard is a real kernel restriction: `TT_FATAL(ternary_a_data_format == in1_data_format)` in `minimal_matmul_program_descriptor.cpp:458`.
- **Cause 2: LoFi helps the fused SwiGLU less than it helped the split up/gate (−10.5 ms of the advantage).**
  - R2-1 shrank the performance advantage from 32.1 to 21.6 ms. The split up/gate 2D matmuls gained ≈ 0.4 ms per layer more from LoFi than the fused SwiGLU does.
  - Together with round 1's total LoFi gain of 31.8 ms, this bounds the SwiGLU's LoFi gain at ≤ ≈ 0.4 ms per call, out of ≈ 1.1 ms of HiFi2 math that LoFi could halve. The fused op is not FPU-bound.
- **Unchanged by precision:**
  - RoPE runs at HiFi4 in both modes (LoFi drops mantissa bits in the ±1 rotation matmul), so R2-2 gained the same in both (−18.5 / −18.0 ms).
  - RMSNorm is HiFi4 in both.
  - SDPA, create heads and the binary ops are DRAM-bound or have no fidelity.
  - In total, 44.9 % of device time (237 ms) is precision-independent.
- The final +19.0 vs the candidate's +12.4 is accuracy-side run noise (530.9 vs 537.4 ms for identical code).
- **What it would take:**
  - **bf16 wo and w_down weights in performance mode** (F4). This is model-only and high confidence. It re-enables the addcmul path under LoFi: expected ≈ 516 ms, ≈ 15 ms faster than accuracy. bf8 weights save nothing on these DRAM-bound ops (4.3 MB per layer).
  - Or relax the ternary_a format check in ttnn, giving the same result with bf8 weights kept.
  - To make performance precision clearly faster than accuracy, use bf8 **activations** on DRAM-bound intermediates:
    - The MLP hidden (SwiGLU output 403 → 214 MB; w_down reads half): ≈ 7–12 ms.
    - The gated ctx / residual stream: more, but the PCC risk is high (performance is at 0.99980 today).
  - RoPE inputs must stay bf16.
  - LoFi on its own will not add much more: no remaining op is FPU-bound.
- A performance-precision profile was again not captured. The per-op split above is inferred from the candidate deltas and the HiFi2 profile.
