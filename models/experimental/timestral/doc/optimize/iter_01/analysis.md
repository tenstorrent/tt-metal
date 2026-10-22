# Round 1 analysis: Timestral-209M, paper shape, one Blackhole p150a (110 worker cores)

Inputs: `bench.json`, `ops_perf_results.csv` (reduced profile: embed + 1 `time` layer + 1 `time_variate` layer + head), `perf_report.*`, `sections/*`, `summary.json`, the round 1 matmul sweeps, and the code in `tt/` after P1.
Full model = embed + 18 x layer_time + 6 x layer_time_variate + head = **457 device ops** (round 0: 513; the 56 standalone SILU/SIGMOID ops are gone).
Conventions are the same as round 0:
- "E" = one bf16 activation 98304 x 768 = 151.0 MB.
- Roofline peaks: HiFi2 359 TFLOPs, 512 GB/s. "FLOPs %" is on the 110-core basis (HiFi2 ≈ 304 TFLOPs) unless marked "/359". tt-perf-report reports FLOPs % relative to the cores an op uses, so for the 80-core gate matmul it prints 38.5 %; on the 110-core basis that op is at 28.0 % (the device perf model's `PM FPU UTIL` agrees: 28.0 %).
- `DRAM %` with `*` is my estimate: logical bytes / 512 GB/s / device time. For matmuls the tt-perf-report value uses the same logical bytes and matches my numbers.
- IDs below are tt-perf-report IDs (= row index in `ops_perf_results.csv` + 2).
- CSV quirk: on program-cache hits, the input-shape columns of Binary rows are stale. For example, the MLP MUL rows 80/105 are logged as (1,1024,96,768)x(1,1,96,768) but are really 98304x2048 x 98304x2048. Elementwise shapes below come from the code; matmul shapes in the CSV are correct.
- Durations are `DEVICE KERNEL DURATION`. `DEVICE FW DURATION` is up to 1.36 ms longer on some ops (e.g. QKV 3.95 vs 2.59 ms) because the next op's firmware starts on idle cores while the previous op finishes. Kernel durations plus the 0.5 µs gaps add up to the measured device time, so the FW excess is overlap, not extra time.

## 1. Headline

| | paper / accuracy | paper / performance | long_context / accuracy | multivariate / accuracy |
|---|---|---|---|---|
| steady wall (median of 5) | **836.4 ms** (runs 833–841) | **817.0 ms** (runs 802–828) | **345.9 ms** | **1209.3 ms** |
| round 0 wall | 1445.3 | 1353.0 | 463.4 | 1797.5 |
| change | −608.9 ms (**−42.1 %**) | −536.0 ms (−39.6 %) | −117.5 ms (−25.4 %) | −588.2 ms (−32.7 %) |
| device (`forward_device` + sync) | **702.0 ms** | **670.2 ms** | 272.9 ms | 1059.3 ms |
| round 0 device | 1307.2 | 1251.3 | 394.4 | 1668.9 |
| change | −605.2 ms (**−46.3 %**) | −581.1 ms (−46.4 %) | −121.5 ms (−30.8 %) | −609.6 ms (−36.5 %) |
| overhead (wall − device) / wall | 134.4 ms = **16.1 %** (r0 9.6 %) | 146.9 ms = 18.0 % (r0 7.5 %) | 73.0 ms = 21.1 % (r0 14.9 %) | 150.0 ms = 12.4 % (r0 7.2 %) |
| ops per forward | 457 (r0 513) | 457 | 457 (same graph) | not profiled |
| (wall − device) / ops | 0.294 ms/op (r0 0.269) | 0.321 ms/op (r0 0.198) | 0.160 ms/op | — |
| prepare / upload / postprocess | 52.0 / 35.8 / 17.7 ms (sum **105.5**; r0 106.7) | 51.3 / 32.1 / 16.3 ms (sum 99.8; r0 94.7) | 42.3 / 18.4 / 12.4 ms (sum 73.1; r0 67.2) | 59.3 / 34.7 / 20.3 ms (sum 114.3; r0 108.5) |
| unattributed (wall − device − split) | 28.9 ms | 47.1 ms | 0.0 ms | 35.7 ms |
| PCC normalized | 0.99993 (r0 0.99995) | 0.99977 (r0 0.99976) | 0.99976 (r0 0.99986) | 0.99994 (r0 0.99995) |

- **The forward is still device-bound.** The measured device time (702.0 ms) matches the profiled kernel sum (700.2 ms, −0.3 %). In-section op-to-op gaps are 0.47–0.61 µs. The host issues a 17-op `time` layer in 2.59 ms (152 µs/op under Tracy) while the device takes 27.2 ms. Host issue is therefore ≈ 70 ms per forward and fully hidden. "(wall − device)/ops" is again fixed per-call host work, not dispatch cost.
- **Host work did not change; only its share did.** prepare + upload + postprocess is 105.5 ms per call, against 106.7 ms in round 0. It is now 12.6 % of wall (round 0: 7.4 %). Host issue would only become exposed if device time fell below ≈ 70 ms.
- **The performance-precision wall numbers are noisy.** Its runs span 802–828 ms, against 833–841 ms for accuracy, and its unattributed remainder is 47 ms against 29 ms. Min-to-min the gap is 30.5 ms (−3.7 %), which agrees with the device delta of 31.8 ms (−4.5 %). See §8.
- **Multivariate: the blocked variate attention was not touched by round 1.** Multivariate − paper device = **357.3 ms** (round 0: 361.7 ms). That is ≈ 59.5 ms extra per `time_variate` layer, against 6.2 ms for the univariate V-path, and it is now 34 % of multivariate device time. It is not profiled (see §7, F9).
- long_context runs at 8.08 µs of device time per token (32 x 1056 tokens) against 7.14 µs per token for paper. Its overhead equals its host split exactly.
- Cold (first-call) time regressed: paper 1.60 → 4.55 s, long_context 0.56 → 13.05 s. Those runs compiled the new 2D/minimal_matmul kernels per shape. Steady-state numbers are unaffected.

## 2. Reconciliation

Estimate from the reduced profile: **700.18 ms**. Measured device: **702.01 ms** (estimate −0.3 %). The profile syncs at every section boundary, which inflates the first gap of each section (353 / 502 / 665 µs) and the head-section gaps (103–181 µs). Those gaps are excluded from device time.

| section | device ms (1 copy) | ops | multiplier | contribution ms | % | round 0 contribution ms |
|---|---|---|---|---|---|---|
| embed | 9.30 | 6 | 1 | 9.30 | 1.3 | 22.09 |
| layer_time | 27.21 | 17 | 18 | 489.79 | 70.0 | 904.40 |
| layer_time_variate | 33.39 | 23 | 6 | 200.35 | 28.6 | 382.60 |
| head | 0.74 | 7 | 1 | 0.74 | 0.1 | 0.90 |
| **total** | | | | **700.18** (457 ops) | 100 | 1309.99 (513 ops) |

`layer_time_variate` − `layer_time` = 6.18 ms (round 0: 13.52 ms). That is the univariate variate sublayer (IDs 96–101: norm, wg + sigmoid, wv, mul, wo, add), i.e. 37.1 ms across the 6 layers.

Per `time` layer: matmuls 11.81 ms (43.4 %), everything else 15.40 ms (56.6 %).

## 3. Category split and matmul utilization

- Full model (sections x multipliers): **Compute 90.1 % / TM 9.9 % / Other 0 %**. Round 0 was 94.7 / 5.3 / 0. TM time is unchanged (69.4 ms vs 69.5 ms); its share rose because compute shrank.
  - Compute 630.8 ms = matmuls 310.8 (44.4 %) + BinaryNg 152.2 (21.7 %) + rotary 95.8 (13.7 %) + RMSNorm 39.1 (5.6 %) + SDPA 32.7 (4.7 %).
  - TM 69.4 ms = create heads 51.6 + concat heads 17.3 + head slice/reshape 0.5.
- Reduced window as printed by tt-perf-report: Compute 75.7 / Other 15.5 / TM 8.8. The "Other" bucket is `MinimalMatmulDeviceOperation`, which tt-perf-report does not classify ("Unclassified operation" warning). Reclassified as compute, the reduced window is 91.2 / 8.8 / 0.
- **Weighted mean matmul FLOPs utilization:**
  - Reduced window: tt-perf-report prints 35.86 % (2D `MatmulDeviceOperation`) and 46.66 % (`MinimalMatmulDeviceOperation`) separately. Combined: 38.1 % (304 basis) / 32.3 % (359 basis).
  - **Full model: 41.1 % (304 basis) / 34.8 % (359 basis).** Round 0: 15.6 % / 13.2 %.
  - The four big matmuls (QKV, up, gate, down): 44.9 % / 38.0 %. The 66 768x768 matmuls (wg, wo, wv): 31.5 % / 26.7 %.
- All matmuls together do 38.8 TFLOP per forward. That is 108.1 ms of math at 359 TFLOPs (127.6 ms at 304), against **310.8 ms** measured (round 0: 818.6 ms). The per-op roofline (max of logical DRAM bytes / 512 GB/s and FLOPs / 359 TFLOPs) sums to **149.0 ms**, so the matmuls are now **2.09x** off their roofline (round 0: 5.5x). The gap is 161.9 ms.
- Every per-token matmul has a logical arithmetic intensity below the HiFi2 ridge of 701 FLOP/B: 382 FLOP/B for 768x768, 555 for 768x2048 and 2048x768, 573 for 768x2304, 611 for 3072x768. Their floor is therefore DRAM traffic, not math. At that floor the best possible FLOPs % is 64.4 % (304) / 54.5 % (359) for 768x768, 93.5 % / 79.2 % for the 2048 shapes and 96.4 % / 81.7 % for QKV.
- Whole graph: **219.0 GB of logical DRAM traffic per forward** (matmuls 76.3 GB, everything else 142.7 GB). The DRAM floor is **427.7 ms**; the forward runs at 1.64x that floor (round 0: 249 GB, 485 ms floor).

**Exit criterion (main matmuls > 65 % FLOPs AND non-matmul ops < 15 % of device time): NO on both counts.**

| condition | target | measured |
|---|---|---|
| main matmul FLOPs % (304 basis) | > 65 % | QKV 44.1, up + SILU 36.5, gate 44.5, down 60.4 (weighted 44.9); 768x768: wg + sigmoid 28.0, wo/wv 35.2 |
| non-matmul share of device time | < 15 % | **55.6 %** (389.4 of 700.2 ms) |

- The 768x768 matmuls cannot reach 65 % at their DRAM roofline (ceiling 64.4 %) with DRAM-interleaved bf16 activations. So the matmul condition is only meaningful for QKV, up, gate and down.
- The non-matmul condition is out of reach with stock ttnn ops. Even at their DRAM floors, SDPA, RMSNorm, create heads, the attention gate MUL and RoPE add up to ≈ 180 ms, which is more than the 149 ms matmul floor. If F1 and F3–F6 in §7 all land (everything except the padding removal), the projected split is ≈ 295 ms matmul and ≈ 205 ms non-matmul, i.e. 41 % non-matmul. Getting under 15 % would need norms, RoPE and head splitting fused into matmul/SDPA kernels.

## 4. Per-op-type table (full model)

| op type | full-model ms | % | count | round 0 ms | round 0 count | change ms |
|---|---|---|---|---|---|---|
| MatmulDeviceOperation (2D multicast) | 228.00 | 32.6 | 106 | 818.56 (all 1D) | 168 | |
| MinimalMatmulDeviceOperation | 82.82 | 11.8 | 62 | — | — | |
| *all matmuls* | *310.82* | *44.4* | *168* | *818.56* | *168* | ***−507.74*** |
| BinaryNg MUL | 96.20 | 13.7 | 55 | 96.23 | 55 | −0.03 |
| RotaryEmbeddingDeviceOperation | 95.84 | 13.7 | 48 | 95.94 | 48 | −0.10 |
| BinaryNg ADD | 56.04 | 8.0 | 57 | 56.16 | 57 | −0.12 |
| NlpCreateHeadsDeviceOperation | 51.65 | 7.4 | 24 | 51.56 | 24 | +0.09 |
| LayerNormDeviceOperation (RMSNorm, HiFi4) | 39.13 | 5.6 | 55 | 39.03 | 55 | +0.10 |
| SDPAOperation | 32.75 | 4.7 | 24 | 32.83 | 24 | −0.08 |
| NLPConcatHeadsDeviceOperation | 17.28 | 2.5 | 24 | 17.51 | 24 | −0.23 |
| Unary SILU | 0 | 0 | 0 | 68.88 | 26 | **−68.88** |
| Unary SIGMOID | 0 | 0 | 0 | 32.81 | 30 | **−32.81** |
| Slice + ReshapeView (head) | 0.47 | 0.07 | 2 | 0.47 | 2 | 0 |
| **total** | **700.18** | 100 | **457** | 1309.99 | 513 | **−609.81** |

Grouped by base op: BinaryNg 152.24 ms (21.7 %, 112 ops); all non-matmul ops 389.36 ms (55.6 %, 289 ops). **Every non-matmul op type has the same per-call time as in round 0.** Round 1 changed only the matmuls and their activations.

Matmuls by role (the round 0 figure includes the standalone unary that followed the matmul):

| role | n | round 0 ms | round 1 ms | per call µs | round 1 engine | speedup |
|---|---|---|---|---|---|---|
| MLP up (+ SILU) | 24 | 192.4 | 66.9 | 2787 | 2D 11x10, SILU fused | 2.88x |
| MLP down | 24 | 191.6 | 40.4 | 1684 | minimal 8/8/8 | 4.74x |
| QKV | 24 | 140.5 | 62.2 | 2593 | 2D 11x10 | 2.26x |
| attn gate (+ SIGMOID), 24 time + 6 variate | 30 | 132.2 | 40.8 | 1361 | 2D 8x10 (80 cores), SIGMOID fused | 3.24x |
| MLP gate | 24 | 126.6 | 54.8 | 2284 | 2D 11x10 | 2.31x |
| attn wo, 24 + 6 | 30 | 98.0 | 32.5 | 1084 | minimal 8/8/8 | 3.02x |
| variate wv | 6 | 19.7 | 6.6 | 1100 | minimal 8/8/8 | 2.98x |
| embed out 3072x768 | 1 | 11.75 | 2.62 | 2624 | minimal 8/8/8 | 4.48x |
| embed hidden 64x3072 (+ SILU) | 1 | 6.39 | 3.01 | 3010 | 2D 11x10, `in0_block_w=2` | 2.12x |
| embed residual 64x768 | 1 | 0.96 | 0.66 | 656 | minimal 8/2/8 | 1.46x |
| head (3 matmuls + SILU) | 3 | 0.38 | 0.22 | — | 2D | 1.71x |
| **total** | **168** | **920.5** | **310.8** | | | **2.96x** |

## 5. Op-to-op map

All ops run on 110 cores (except the 80-core gate matmuls and the 64-core head norm), DRAM interleaved, bf16, HiFi2 for matmuls. The configs come from `tt/common.py::linear` → `tt/program_configs.py`:
- "2D" = `MatmulMultiCoreReuseMultiCastProgramConfig` with `in0_block_w=12`, `out_block_h=11`, `per_core_M=308` (= 28 x 11), `out_block_w = per_core_N`, `out_subblock_h=1`. The subblock height of 1 is forced because 11 is prime; the sweep found `obh=11` best anyway.
- "minimal" = `ttnn.experimental.minimal_matmul` with M/K/N blocks 8/8/8, subblock 1x8, grid 11x10.

Bound: tt-perf-report for matmuls; `DRAM*`/`SLOW*` is my classification (estimated DRAM % at least 65 % or not).

### One `time` layer (IDs 66–82, 27.21 ms; round 0 50.24 ms)

| ID | op | shape / M x K x N | µs | bound | DRAM % | FLOPs % (304 / 359) | config | source |
|---|---|---|---|---|---|---|---|---|
| 66 | rms_norm (attn) | 98304x768 | 729 | DRAM* | 81* | — | default, HiFi4 | attention.py:64 → common.py:141 |
| 67 | linear wg + SIGMOID | 98304x768x768 | 1361 | SLOW | 43.5 | 28.0 / 23.7 | 2D 8x10, per_core_N=3, sb 1x3 | attention.py:65 → common.py:128 |
| 68 | linear wqkv | 98304x768x2304 | 2592 | SLOW | 45.8 | 44.1 / 37.4 | 2D 11x10, per_core_N=7, sb 1x7 | attention.py:141 |
| 69 | split_qkv_and_split_heads | 98304x2304 → 3 x (1024,12,96,64) | 2152 | DRAM* | 82* | — | | attention.py:83 (via :142) |
| 70 | rotary_embedding q | (1024,12,96,64) | 1990 | SLOW* | 30* | — | | attention.py:122 (via :143) |
| 71 | rotary_embedding k | (1024,12,96,64) | 2004 | SLOW* | 29* | — | | attention.py:122 (via :143) |
| 72 | SDPA causal | q/k/v (1024,12,96,64) | 1365 | DRAM* | 86* | 3.5 (math) | chunk 96 | attention.py:99 (via :144) |
| 73 | concatenate_heads | → 98304x768 | 718 | DRAM* | 82* | — | | attention.py:113 |
| 74 | MUL ctx * gate | 98304x768 | 1012 | DRAM* | 87* | — | SFPU mul | attention.py:73 |
| 75 | linear wo | 98304x768x768 | 1090 | SLOW | 54.3 | 35.0 / 29.6 | minimal | attention.py:76 → common.py:120 |
| 76 | ADD residual (attn) | 98304x768 | 1005 | DRAM* | 88* | — | | transformer.py:28 → common.py:154 |
| 77 | rms_norm (MLP) | 98304x768 | 724 | DRAM* | 82* | — | default, HiFi4 | mlp.py:31 |
| 78 | linear w_up + SILU | 98304x768x2048 | 2785 | SLOW | 39.0 | 36.5 / 30.9 | 2D 11x10, per_core_N=6, sb 1x6 | mlp.py:32 |
| 79 | linear w_gate | 98304x768x2048 | 2286 | SLOW | 47.6 | 44.5 / 37.7 | 2D 11x10, per_core_N=6, sb 1x6 | mlp.py:33 |
| 80 | MUL gate * up | 98304x2048 | 2705 | DRAM* | 87* | — | SFPU mul | mlp.py:35 |
| 81 | linear w_down | 98304x2048x768 | 1696 | SLOW | 64.1 | 60.0 / 50.8 | minimal | mlp.py:38 |
| 82 | ADD residual (MLP) | 98304x768 | 996 | DRAM* | 89* | — | | transformer.py:29 |

Rows 78 and 79 run the same 2D config on the same shape. The only difference is the fused SILU, which costs **+499 µs** in-kernel. The round 0 standalone SILU was 2755 µs.

### One `time_variate` layer (IDs 85–107, 33.39 ms; round 0 63.77 ms)

| ID | op | shape / M x K x N | µs | bound | DRAM % | FLOPs % (304 / 359) | source |
|---|---|---|---|---|---|---|---|
| 85 | rms_norm (time attn) | 98304x768 | 718 | DRAM* | 82* | — | attention.py:64 |
| 86 | linear wg + SIGMOID (time) | 98304x768x768 | 1365 | SLOW | 43.4 | 27.9 / 23.7 | attention.py:65 |
| 87 | linear wqkv | 98304x768x2304 | 2594 | SLOW | 45.7 | 44.1 / 37.4 | attention.py:141 |
| 88 | split_qkv_and_split_heads | → 3 x (1024,12,96,64) | 2153 | DRAM* | 82* | — | attention.py:83 |
| 89 | rotary_embedding q | (1024,12,96,64) | 1991 | SLOW* | 30* | — | attention.py:122 |
| 90 | rotary_embedding k | (1024,12,96,64) | 1999 | SLOW* | 30* | — | attention.py:122 |
| 91 | SDPA causal | (1024,12,96,64) | 1363 | DRAM* | 86* | — | attention.py:99 |
| 92 | concatenate_heads | → 98304x768 | 724 | DRAM* | 82* | — | attention.py:113 |
| 93 | MUL ctx * gate | 98304x768 | 1001 | DRAM* | 88* | — | attention.py:73 |
| 94 | linear wo (time) | 98304x768x768 | 1073 | SLOW | 55.2 | 35.5 / 30.1 | attention.py:76 |
| 95 | ADD residual | 98304x768 | 996 | DRAM* | 89* | — | transformer.py:28 |
| 96 | rms_norm (variate attn) | 98304x768 | 723 | DRAM* | 82* | — | attention.py:64 |
| 97 | linear wg + SIGMOID (variate) | 98304x768x768 | 1359 | SLOW | 43.6 | 28.1 / 23.8 | attention.py:65 |
| 98 | linear wv (univariate V-path) | 98304x768x768 | 1100 | SLOW | 53.8 | 34.7 / 29.4 | attention.py:69 |
| 99 | MUL v * gate | 98304x768 | 1004 | DRAM* | 88* | — | attention.py:73 |
| 100 | linear wo (variate) | 98304x768x768 | 1077 | SLOW | 55.0 | 35.4 / 30.0 | attention.py:76 |
| 101 | ADD residual | 98304x768 | 1006 | DRAM* | 88* | — | transformer.py:28 |
| 102 | rms_norm (MLP) | 98304x768 | 718 | DRAM* | 82* | — | mlp.py:31 |
| 103 | linear w_up + SILU | 98304x768x2048 | 2792 | SLOW | 39.0 | 36.4 / 30.9 | mlp.py:32 |
| 104 | linear w_gate | 98304x768x2048 | 2280 | SLOW | 47.7 | 44.6 / 37.8 | mlp.py:33 |
| 105 | MUL gate * up | 98304x2048 | 2708 | DRAM* | 87* | — | mlp.py:35 |
| 106 | linear w_down | 98304x2048x768 | 1648 | DRAM | 66.0 | 61.7 / 52.3 | mlp.py:38 |
| 107 | ADD residual | 98304x768 | 999 | DRAM* | 89* | — | transformer.py:29 |

Configs are as in the `time` layer: 2D for wg/wqkv/up/gate, minimal for wo/wv/down.

Embed (IDs 58–63, 9.30 ms; round 0 22.09 ms):

- 58: hidden 98304x64x3072 + bias + SILU, 3010 µs. 2D 11x10, `in0_block_w=2` (K is only 2 tiles), out block 11x9, subblock 1x3. DRAM 40.0 %, FLOPs 4.2 % (residual_block.py:43).
- 59: output 98304x3072x768 + bias, 2624 µs. Minimal; DRAM 56.5 %, FLOPs 58.1 % (residual_block.py:44).
- 60: residual 98304x64x768 + bias, 656 µs. Minimal 8/2/8 (residual_block.py:46).
- 61: ADD main + skip, 994 µs (residual_block.py:47).
- 62: MUL keep, broadcast (1,96,768), 1011 µs (model.py:198).
- 63: ADD query, 1004 µs (model.py:200).
- 62 and 63 run at 58 %* on logical bytes because the broadcast operand saves no time.

Head (IDs 110–116, 0.74 ms): slice 270 µs (model.py:211), reshape copy 204 µs (:213), rms_norm 21 µs on 64 cores (:214), then the 2D matmuls hidden + SILU 103 µs, output 84 µs, residual 33 µs, and an ADD of 24 µs.

## 6. Top 10 op roles (full model, with multiplicity)

The lower bound (LB) per call is max(FLOPs / 359 TFLOPs, logical bytes / 512 GB/s), where bytes = input + weight + output in bf16. Headroom = (actual − LB) x n.

| # | role | n | full ms | per call µs | M x K x N / shape | cores | DRAM % | FLOPs % (304 / 359) | limit hit | LB/call µs | headroom ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rotary q + k | 48 | 95.8 | 1997 | (1024,12,96,64) | 110 | 29.5* | — | neither: latency (serialized single-tile reads) | 590 (DRAM) | **67.5** |
| 2 | MLP up 768x2048 + SILU (2D) | 24 | 66.9 | 2787 | 98304x768x2048 | 110 | 39.0 | 36.5 / 30.9 | neither: 2D pipeline + in-kernel SILU (+500 µs) | 1087 (DRAM; math 861) | **40.8** |
| 3 | MLP MUL gate * up | 24 | 64.9 | 2706 | 98304x2048 | 110 | 87.2* | — | DRAM | 2359 | 8.3 |
| 4 | QKV 768x2304 (2D) | 24 | 62.2 | 2593 | 98304x768x2304 | 110 | 45.8 | 44.1 / 37.4 | neither | 1187 (DRAM; math 969) | **33.7** |
| 5 | MLP gate 768x2048 (2D) | 24 | 54.8 | 2284 | 98304x768x2048 | 110 | 47.6 | 44.5 / 37.7 | neither | 1087 (DRAM; math 861) | **28.7** |
| 6 | residual ADD | 54 | 54.0 | 1000 | 98304x768 | 110 | 88.4* | — | DRAM | 885 | 6.2 |
| 7 | create heads (split qkv) | 24 | 51.6 | 2152 | 98304x2304 → 3 x (1024,12,96,64) | 110 | 82.2* | — | DRAM | 1770 | 9.2 |
| 8 | attn gate 768x768 + SIGMOID (2D), 24 time + 6 variate | 30 | 40.8 | 1361 | 98304x768x768 | **80** | 43.5 | 28.0 / 23.7 | neither: 80 cores + in-kernel sigmoid (≈ +300 µs) | 592 (DRAM; math 323) | **23.1** |
| 9 | MLP down 2048x768 (minimal) | 24 | 40.4 | 1684 | 98304x2048x768 | 110 | 64.6 | 60.4 / 51.1 | neither, just under DRAM (ID 106 crosses 65 %) | 1087 (DRAM; math 861) | 14.3 |
| 10 | rms_norm | 55 | 39.1 | 711 | 98304x768 | 110 | 81.4* | — | DRAM | 579 | 7.3 |

Together these ten roles are 570.5 ms (81.5 %). Matmul gaps per call: up 2.56x LB, attn gate 2.30x, QKV 2.18x, gate 2.10x, down 1.55x. Next in line:

| role | n | full ms | per call µs | headroom ms | notes |
|---|---|---|---|---|---|
| SDPA | 24 | 32.7 | 1365 | 4.4 | 86 %* DRAM; math 3.5 % (perf model) |
| attn wo 768x768 (minimal) | 30 | 32.5 | 1084 | 14.8 | DRAM 54.6 %, FLOPs 35.2 %; 1.83x LB |
| attn MUL (ctx or v) * gate | 30 | 30.2 | 1008 | 3.7 | 88 %* DRAM |
| concat heads | 24 | 17.3 | 720 | 3.1 | 82 %* DRAM |
| variate wv 768x768 (minimal) | 6 | 6.6 | 1100 | 3.0 | |
| embed (6 ops) | 1 | 9.3 | — | 4.2 | hidden 2D at 40 % DRAM (LB 1205 µs) |

Rotary was the 7th-largest role in round 0 and is now the largest. Its per-call time is unchanged at 2.0 ms.

## 7. Findings, ranked by estimated full-model savings

Savings overlap and are not additive. All estimates are full-model device ms at the paper shape, accuracy precision.

**F1. Fused SwiGLU (round 0 P2): `minimal_matmul(fuse_swiglu=True)` on a tile-pair-interleaved [up | gate] weight. Savings ≈ 70–100 ms and −48 ops. Confidence: medium.**

- Evidence rows: 78–80 and 103–105. Today the MLP front half is up + SILU 2787 µs + gate 2284 µs + MUL 2706 µs = **7.78 ms per layer, 186.6 ms over 24 layers (26.7 % of device time)**. It moves 2.32 GB per layer: xn is read twice, two 403 MB outputs are written and then read back by the MUL, and the MUL writes 403 MB.
- The fused op reads xn once (151 MB), reads a 768x4096 weight (6.3 MB) and writes the 2048-wide product once (403 MB): 560 MB, a DRAM floor of 1.09 ms. It does 618.5 GFLOP, a 1104 FLOP/B intensity that is above the HiFi2 ridge, so it is compute-bound. At the minimal_matmul rates measured in this profile (between 44 % and 60–62 % of 304 TFLOPs, the latter being ID 81/106) it takes 3.4–4.6 ms, plus any unhidden SFPU epilogue. The kernel runs the SwiGLU inside the K loop when there is no bias and `subblock_w` is even (`minimal_matmul_program_descriptor.cpp:510`).
- Constraints: N, N per core and N_block must all be even (`:345-351`), and it cannot be combined with `fused_activation` or addcmul. Prior art in this tree: `models/experimental/qwen_image_2_1/tt/dit.py:591`, `models/demos/qwen38_27b_qb2/tt/decoder.py:569`, and the `models/tt_dit/utils/tensor.py:21` `prepare_for_fused_swiglu` weight layout. Remember to keep the RMSNorm gain folded into both halves.
- Side effect: this is the first compute-bound op in the graph, so LoFi (performance precision) would gain ≈ 1.5 ms/layer on it (§8).

**F2. Remove per-row tile padding outside attention (round 0 P6). Savings ≈ 40–90 ms today, ≈ 0–50 ms once F1/F4/F6 land. Confidence: low. Complexity: high. Do it last.**

- Per-token ops outside the padded attention core (everything except create heads, rotary, SDPA and concat heads, which total 197.5 ms) take 501.9 ms. 31.25 % of that is pad work: **157 ms gross**. 1024 x 66 = 2112 tiles is tile-aligned, so the flat ops could run on 67584 rows (`per_core_M` 308 → 212).
- If the stream is unpacked before wqkv (768 wide) rather than after it (2304 wide), wqkv stays padded and the gross falls to 137 ms.
- Cost: an unpack of xn and a pack of ctx per time-attention sublayer, 48 tiled row-granular reshapes of ≈ 255 MB each. That is ≥ 0.5 ms each at peak, realistically 1–2 ms, so **48–96 ms**. Garbage in the pad rows is safe for causal SDPA only if it is finite.
- Gate: a stand-alone tiled-reshape micro-benchmark at the real shape; proceed only if a pair costs ≤ 1.5 ms. After F1/F4/F6 shrink the flat work to ≈ 365 ms, the gross drops to ≈ 95 ms.

**F3. RoPE through `ttnn.experimental.rotary_embedding_llama` (round 0 P3). Savings ≈ 45–60 ms. Confidence: medium.**

- Evidence rows: 70, 71, 89, 90. 48 calls of 2.0 ms each = 95.8 ms (13.7 %), at 29.5 %* of DRAM peak and unchanged since round 0. The current reader issues 4 serialized single-tile reads per output tile, and the input is read twice.
- `rotary_embedding_hf` has the same 4-read pattern (`reader_rotary_embedding_hf_interleaved.cpp:68-108`), so switching to it does not help.
- The llama prefill path accepts batch > 1, interleaved DRAM input and head-broadcast cos/sin (prefill branch of `validate` in `rotary_embedding_llama_device_operation.cpp`). It splits work over batch (1024 → about 10 per core). Its reader fetches each input tile once (2 tiles per barrier) and cos/sin once per batch (`reader_rotary_embedding_llama_interleaved_start_id.cpp:71-160`). Expected 0.7–1.0 ms per call against a 0.59 ms floor.
- Requirements: permute the Q and K output rows of `wqkv` at load time into interleaved-pair order. This is exact, because q·k is invariant under the same head_dim permutation applied to both, and V is untouched. Also build llama-style cos/sin tables (identity cos = 1, sin = 0 on the partial-RoPE dims) and a (1,1,32,32) trans_mat; all inputs must be bf16.
- Going below that would mean fusing RoPE into create heads, which needs a kernel change. It is worth up to 95.8 ms.

**F4. Fuse the residual ADD into wo and w_down with `ttnn.experimental.dit_minimal_matmul_addcmul_fused` (out = residual + 1.0 x (A @ W) x ones[1, N]). Net savings ≈ 25–35 ms and −54 ops. Confidence: medium.**

- Evidence: all 54 residual ADDs (IDs 76, 82, 95, 101, 107; 1000 µs each, 54.0 ms, 88 %* DRAM) directly follow a minimal_matmul (30 wo + 24 down), the engine this op wraps. It accepts the residual at full [M, N] and the multiplier at [1, N] (`minimal_matmul_device_operation.cpp:201-244`). It is used in `models/tt_dit/layers/linear.py:519` and `qwen_image_2_1/tt/dit.py:371`.
- Cost: the matmul reads one extra 151 MB tensor. At minimal_matmul's observed 280–330 GB/s that is +0.3–0.55 ms per call, ≈ 26 ms in total.
- It removes 302 MB per residual from the DRAM floor (16.3 GB, ≈ 32 ms of floor).

**F5. SDPA `output_concat_heads=True`. Savings ≈ 15–17 ms and −24 ops. Confidence: high.**

- Evidence rows: 73 and 92, concat heads 24 x 720 µs = 17.3 ms. SDPA can write [b x 1 x s x nqh·dh] directly (`sdpa_nanobind.cpp:351`). This is supported for plain non-chunked SDPA, which is exactly the time-attention call. Prior art: `models/tt_dit/models/vae/minimax_h3/decoder_minimax_h3.py:229`.
- The output bytes are unchanged; only the tile placement differs.

**F6. Move matmul-epilogue activations into the consumer MUL (`ttnn.mul(..., input_tensor_b_activations=[SIGMOID or SILU])`) and run wg on minimal_matmul. Savings ≈ 8 ms (sigmoid part), or ≈ 20 ms if F1 is not done. Confidence: medium-high.**

- Evidence: the in-kernel SFPU activation is expensive in both matmul engines because it serializes on the math thread.
  - SILU: +499 µs per call (ID 78 vs 79; 103 vs 104).
  - SIGMOID: the round 1 performance sweep has the same 2D config (`2d[8x10,obh=11,ibw=12,sb=1x3]`) at 1.043 ms without it (proj768) and 1.360 ms with it (gate_sigmoid); minimal_matmul with fused sigmoid took 1.71 ms against 1.48 ms for 2D (`program_configs.py` docstring).
- The MULs are DRAM-bound with ≈ 2000 cycles of slack per tile: 670 tiles per core in 1.0 ms for the attention MUL, 1787 tiles per core in 2.7 ms for the MLP MUL. A precise sigmoid/SILU costs ≈ 380–470 cycles per tile, inferred from the in-kernel deltas, so the MUL should absorb it with no time increase.
- wg then runs on 110 cores as a minimal_matmul (≈ 1084 µs, like wo) instead of 1361 µs on 80 cores: 30 x 277 µs = 8.3 ms.
- The SILU half is subsumed by F1.

**F7. Remaining matmul-config gap. Savings ≤ 10–15 ms from more config search. Confidence: low.**

- After F1/F6, the matmuls still off their roofline are QKV (headroom 33.7 ms), wo/wv (17.8 ms) and down (14.3 ms).
- The round 1 sweep has plateaued: the top QKV candidates are within 1–2 % (2D `obh=11,ibw=12` 2.277 ms vs minimal 2.251 ms in the performance sweep), and the minimal 768x768 variants are within 2 %. Larger minimal blocks (16/8/8) fail circular-buffer allocation (`dataflow_buffer.cpp:3087` TT_THROW) on every shape with bf16 weights. With bf8 weights they run but only tie (QKV 2.251 ms).
- Beyond tuning, the levers are structural: L1 inputs via M-chunking, or bf8 activations.

**F8. Small cleanups, ≈ 2–3 ms in total. Confidence: medium.**

- Embed: fold main + skip into the embed out matmul via addcmul (≈ 0.6 ms). Replace MUL keep + ADD query with one `ttnn.addcmul(add, embedded, keep)` (≈ 1.0 ms).
- Head: the slice + reshape copy (0.47 ms) could read only the 2 decoder rows.
- Per-op tuning headroom elsewhere: rms_norm (≤ 7 ms, already 81 %), create heads (≤ 9 ms, 82 %) and SDPA (≤ 4 ms, 86 %) are near their floors.

### Re-evaluation of the round 0 deferred and unapproved proposals

| proposal | round 0 estimate | worth now | why |
|---|---|---|---|
| P2 fused SwiGLU (`fuse_swiglu=True`) | 200–280 ms (overlapping P1) | **70–100 ms** (F1) | P1 already took the re-read waste, but the MLP front half is still the largest block (186.6 ms) and the fusion removes 1.76 GB per layer of traffic. Top candidate. |
| P3 `rotary_embedding_llama` | 40–65 ms | **45–60 ms** (F3) | Unchanged in ms, but it is now the largest single role (13.7 %). |
| P4 host prepare/upload/postprocess | 30–60 ms wall | **30–60 ms wall** (no device effect) | 105.5 ms per call, now 12.6 % of wall instead of 7.4 %. Still not profiled at function level; run cProfile first. |
| P5 pack wg into wqkv; variate wg + wv | 15–30 ms (after P1) | **≈ 0 — drop** | In0 is now read once, so merging only saves one 151 MB read of xn (≈ 0.3–0.5 ms) per merge. Each merge needs a slice/split (≥ 0.6 ms, 302 MB), loses the per-role fused activation, and `split_query_key_value_and_split_heads` needs exactly 3·D columns. |
| P6 drop per-row padding outside attention | 150–290 ms | **40–90 ms now; 0–50 ms after F1/F4/F6** (F2) | Everything left is near-DRAM-bound and scales with M, but the re-layout cost does not shrink. Gate on a tiled-reshape micro-benchmark. |
| Precision per tensor group (bf8 / LoFi) | retry after P1/P2 | **performance precision only: LoFi ≈ 35 ms after F1; bf8 activations ≈ 20–40 ms** | LoFi only helps compute-bound ops; today none are (§8). bf8 *weights* are 0.4 GB of 219 GB of traffic. bf8 *activations* on DRAM-bound intermediates (for example the MLP hidden after F1) halve their bytes, but `rotary_embedding_llama` needs bf16 inputs. Not for accuracy precision. |
| L1-resident M-chunks | later round | **≈ 50–100 ms, after F1/F4/F5** | Those three cut the logical traffic from 219 GB to ≈ 153 GB (floor 428 → ≈ 299 ms). Keeping, say, xn → fused SwiGLU → down or the attention tail in L1 per M-chunk is the next structural step. Very high complexity, low confidence. |
| Tracing | no device gain | **no device gain** | Host issue (≈ 70 ms per forward) is hidden behind 702 ms of device time. The "Running with tracing could save 2048 µs" advice comes from the sync-inflated section-boundary gaps. |

Other ideas I checked and rejected:
- Fusing the gate MUL into the wo input: no ttnn matmul has an elementwise prologue on in0, and addcmul applies on the output side. It would need a kernel change (≈ 18 ms).
- `rms_norm(residual_input_tensor=)`: it still does not return the sum, so the norm cannot replace the residual ADD.
- `rotary_embedding_hf`: same reader pattern as today's op.

**F9 (multivariate only). The blocked variate attention costs +357 ms of device time over paper. Worth ≥ 200 ms on multivariate. Not profiled; confidence low.**

- `_variate_context` (attention.py:147-164) does the following per time_variate layer:
  - permutes the 453 MB QKV to slot-major;
  - reshapes it to (S·B, V=8, 3D), so each group of 8 rows sits in a 32-row tile, a **4x padding inflation** for every downstream op;
  - optionally fills the tile padding;
  - splits heads and runs SDPA on (12288, 12, 32, 64);
  - permutes back.
- Next round should profile the multivariate shape. Two variate-specific candidates: SDPA `output_concat_heads` here too, and a layout that keeps V on a non-tiled axis.

**Structural floor.** At 219.0 GB per forward the current graph cannot go below 428 ms even with every op at its roofline. F1 + F4 + F5 cut that to ≈ 153 GB (≈ 299 ms). With F1 and F3–F6 at their midpoints, device time would be ≈ 700 − 85 − 52 − 30 − 16 − 8 ≈ **510 ms**, with ≈ 330 ops.

## 8. Comparison with the performance precision (bf8 weights, LoFi)

- Device: 702.0 → 670.2 ms (**−31.8 ms, −4.5 %**). Wall: 836.4 → 817.0 ms (−19.4 ms, −2.3 %). The wall delta understates the gain because the performance runs are noisier: they span 802–828 ms and leave 47 ms unattributed, against 29 ms for accuracy. Min-to-min the gap is 30.5 ms (−3.7 %), which matches the device delta. Host prepare/upload/postprocess differ by 5.7 ms with identical host code, which is noise. So the real effect is ≈ 4.5 %, not 2 %.
- **55.6 % of device time (389 ms) is precision-independent.**
  - RMSNorm always runs at HiFi4 (`norm_kernel_config`).
  - Rotary has its own fixed HiFi4 config.
  - Binary ops, create heads and concat heads have no math fidelity.
  - SDPA does switch to LoFi, but it is DRAM-bound (86 %, math 3.5 %), so it gains ≈ 0.
- **bf8 weights do not matter.** Weights are ≈ 0.4 GB of the 219 GB of logical traffic per forward, and every per-token matmul's traffic is dominated by activations.
- **LoFi only trims the part of matmul time where the FPU is on the critical path.** HiFi2 math is 127.6 ms on the 110-core basis. If it were fully serialized with data movement, LoFi would save up to half of it, 63.8 ms. The observed 31.8 ms means only ≈ 50 % of the math is exposed. In round 0, 88 % was exposed (56 of 64 ms).
- The round 1 engines overlap math with data movement better, and they are now limited by data movement and pipeline latency: DRAM 39–66 % of peak, FLOPs 28–62 %. Their logical intensity (382–611 FLOP/B) is below even the HiFi2 ridge of 701 FLOP/B (LoFi ridge 1404), so they stay DRAM-floor-bound at either fidelity.
- **What would make performance precision matter:**
  - F1, which creates a compute-bound op at 1104 FLOP/B. LoFi would roughly halve its ≈ 3.4 ms of math, ≈ 1.5 ms per layer or ≈ 35 ms in total.
  - bf8 *activations* (not weights) on DRAM-bound intermediates.
  - A performance-precision profile was not captured this round, so the per-op split above is inferred from the device delta and the HiFi2 profile.
