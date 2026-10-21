# Round 0 analysis: Timestral-209M, paper shape, one Blackhole p150a (110 worker cores)

Inputs: `bench.json`, `ops_perf_results.csv` (reduced profile: embed + 1 `time` layer + 1 `time_variate` layer + head), `perf_report.*`, `sections/*`, `summary.json`.
Full model = embed + 18 x layer_time + 6 x layer_time_variate + head = 513 device ops.
Conventions: "E" = one bf16 activation 98304 x 768 = 151.0 MB. Roofline peaks follow the instructions (HiFi2 359 TFLOPs, 512 GB/s). The tt-perf-report `FLOPs %` column uses the 110-core peak (HiFi2 ≈ 304 TFLOPs; I checked this: 35.44 TFLOPs / 11.65 % = 304). `DRAM %` with `*` is my estimate: logical bytes / 512 GB/s / device time.
CSV quirk: on program-cache hits, the input-shape columns of Unary and Binary rows are stale. For example, row 88 SILU is logged as 98304x3072 but is really 98304x2048, and the MUL rows 83/90/104/111/118 are logged as (1,1024,96,768)x(1,1,96,768). Shapes below come from the source code. Matmul shapes in the CSV are correct.

## 1. Headline

| | paper / accuracy | paper / performance |
|---|---|---|
| steady wall (median of 5) | **1445.3 ms** (runs 1409–1449, bimodal) | **1353.0 ms** (runs 1338–1371) |
| device = `forward_device` + sync | **1307.2 ms** | **1251.3 ms** |
| overhead (wall − device) / wall | 138.1 ms = **9.6 %** | 101.7 ms = **7.5 %** |
| ops per forward (summary.json estimate) | 513 | 513 |
| (wall − device) / ops | 0.269 ms/op | 0.198 ms/op |
| prepare / upload / postprocess | 52.4 / 36.4 / 17.9 ms (sum 106.7) | 46.3 / 31.5 / 16.9 ms (sum 94.7) |
| unattributed (wall − split sum) | 31.4 ms | 7.0 ms |

- **The forward is device-bound, not host-bound.** The forward wall time (1307.2 ms) equals the profiled kernel sum (1310.0 ms, +0.2 %). In-section op-to-op gaps are 0.47–0.60 µs. The host issues a whole 19-op layer in 2.7 ms while the device takes 50.2 ms; host issue is about 150–200 µs/op under Tracy, so roughly 80–100 ms per forward, all hidden. Consequently "(wall − device)/ops" is **not** a dispatch cost. It is fixed per-call host work that does not scale with op count.
- **What the non-device time is:**
  - prepare: host torch work (causal per-patch loc/scale in float64, asinh, building the `[values, missing]` tokens, RoPE tables, masks).
  - upload: host build of the (98304, 64) token tensor, fp32 → bf16, PCIe transfer, on-device tilize (~0.2 ms), cos/sin upload.
  - postprocess: `to_torch` of the (2048, 672) head output, then host monotone quantiles and sinh de-normalization.
  - The 31 ms unattributed in accuracy mode is run-to-run variance, not a missing stage. Two steady runs took ≈1.41 s and three ≈1.45 s, and the separately measured split totals 1413.9 ms, which matches the fast mode.
- Other shapes (not profiled this round): long_context 463.4 ms wall / 394.4 ms device (14.9 % overhead). multivariate 1797.5 / 1668.9 ms, which is **+361.7 ms device** versus paper at the same token count. The blocked variate attention therefore costs ≈ 60 ms more per `time_variate` layer than the univariate V-path.

## 2. Reconciliation

Estimate from the reduced profile: **1310.0 ms**. Measured device: **1307.2 ms** (+0.2 %). The profile syncs at every section boundary, which inflates the first gap of each section (385 / 680 / 721 µs) and the head-section gaps (71–482 µs). Those gaps are excluded from device time.

| section | device ms (1 copy) | ops | multiplier | contribution ms | % |
|---|---|---|---|---|---|
| embed | 22.09 | 7 | 1 | 22.09 | 1.7 |
| layer_time | 50.24 | 19 | 18 | 904.40 | 69.0 |
| layer_time_variate | 63.77 | 26 | 6 | 382.60 | 29.2 |
| head | 0.90 | 8 | 1 | 0.90 | 0.1 |
| **total** | | | | **1309.99** (513 ops) | 100 |

`layer_time_variate` − `layer_time` = 13.52 ms. That is the univariate variate sublayer (rows 107–113: norm, wg, sigmoid, wv, mul, wo, add), i.e. 81.1 ms across the 6 layers.

## 3. Category split and matmul utilization

- Full model (sections x multipliers): **Compute 94.7 % / TM 5.3 % / Other 0 %**. Reduced window as reported by tt-perf-report: 95.4 / 4.6 / 0.
- Weighted mean matmul FLOPs utilization:
  - **14.47 %** for the reduced window (tt-perf-report, 110-core HiFi2 basis).
  - Full model: **15.6 %** on the 304 TFLOPs basis, **13.2 %** on the 359 TFLOPs basis.
- All matmuls together do 38.8 TFLOP per forward. That is 108 ms of math at 359 TFLOPs (128 ms at 304), against **818.6 ms** measured.
- Every `MatmulDeviceOperation` is classified SLOW (DRAM 12.6–21 %, FLOPs 3.3–19.7 %).

## 4. Per-op-type table (full model)

| op type | full-model ms | % | count |
|---|---|---|---|
| MatmulDeviceOperation | 818.56 | 62.5 | 168 |
| BinaryNg MUL | 96.23 | 7.3 | 55 |
| RotaryEmbeddingDeviceOperation | 95.94 | 7.3 | 48 |
| Unary SILU | 68.88 | 5.3 | 26 |
| BinaryNg ADD | 56.16 | 4.3 | 57 |
| NlpCreateHeadsDeviceOperation | 51.56 | 3.9 | 24 |
| LayerNormDeviceOperation (RMSNorm, HiFi4) | 39.03 | 3.0 | 55 |
| SDPAOperation | 32.83 | 2.5 | 24 |
| Unary SIGMOID | 32.81 | 2.5 | 30 |
| NLPConcatHeadsDeviceOperation | 17.51 | 1.3 | 24 |
| Slice + ReshapeView (head) | 0.47 | 0.04 | 2 |
| **total** | **1309.99** | 100 | **513** |

Grouped by base op: BinaryNg 152.39 ms (11.6 %, 112 ops); Unary 101.69 ms (7.8 %, 56 ops).

## 5. Op-to-op map

Bound for matmuls comes from tt-perf-report. `DRAM*`/`SLOW*` is my classification (estimated DRAM % above or below 65 %). All ops run on 110 cores, DRAM interleaved, bf16. Every matmul uses `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0=0`, `per_core_M=28` and `fused_activation=nullopt`. The out_block shapes are listed in §7 F1.

### One `time` layer (rows 74–92, 50.24 ms)

| ID | op | M x K x N | µs | bound | cores | DRAM % | FLOPs % | in0_bw | subblock | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 74 | rms_norm (attn) | — | 720 | DRAM* | 110 | 82* | — | — | — | attention.py:64 → common.py:112 |
| 75 | linear wg (gate) | 98304x768x768 | 3272 | SLOW | 110 | 18.1 | 11.7 | 2 | 4x2 | attention.py:65 → common.py:98 |
| 76 | SIGMOID (unfused activation of 75) | — | 1093 | SLOW* | 110 | 54* | — | — | — | attention.py:65 → matmul.cpp:380 (trailing unary_chain) |
| 77 | linear wqkv | 98304x768x2304 | 5852 | SLOW | 110 | 20.3 | 19.5 | 2 | 7x1 | attention.py:141 |
| 78 | split_qkv_and_split_heads | — | 2148 | DRAM* | 110 | 82* | — | — | — | attention.py:83 (via :142) |
| 79 | rotary_embedding q | — | 1993 | SLOW* | 110 | 30* | — | — | — | attention.py:122 (via :143) |
| 80 | rotary_embedding k | — | 2007 | SLOW* | 110 | 29* | — | — | — | attention.py:122 (via :143) |
| 81 | SDPA causal, q/k chunk 32 | — | 1368 | DRAM* | 110 | 86* | — | — | — | attention.py:99 (via :144) |
| 82 | concatenate_heads | — | 730 | DRAM* | 110 | 81* | — | — | — | attention.py:113 |
| 83 | MUL ctx*gate | — | 992 | DRAM* | 110 | 89* | — | — | — | attention.py:73 |
| 84 | linear wo | 98304x768x768 | 3198 | SLOW | 110 | 18.5 | 11.9 | 2 | 4x2 | attention.py:76 |
| 85 | ADD residual (attn) | — | 1006 | DRAM* | 110 | 88* | — | — | — | transformer.py:28 → common.py:121 |
| 86 | rms_norm (MLP) | — | 723 | DRAM* | 110 | 82* | — | — | — | mlp.py:31 |
| 87 | linear w_up | 98304x768x2048 | 5277 | SLOW | 110 | 20.6 | 19.3 | 2 | 2x4 | mlp.py:32 |
| 88 | SILU (unfused activation of 87) | — | 2807 | SLOW* | 110 | 56* | — | — | — | mlp.py:32 → matmul.cpp:380 |
| 89 | linear w_gate | 98304x768x2048 | 5308 | SLOW | 110 | 20.5 | 19.2 | 2 | 2x4 | mlp.py:33 |
| 90 | MUL gate*up | — | 2730 | DRAM* | 110 | 86* | — | — | — | mlp.py:35 |
| 91 | linear w_down | 98304x2048x768 | 8017 | SLOW | 110 | 13.6 | 12.7 | 2 | 4x2 | mlp.py:38 |
| 92 | ADD residual (MLP) | — | 1001 | DRAM* | 110 | 88* | — | — | — | transformer.py:29 → common.py:121 |

### One `time_variate` layer (rows 95–120, 63.77 ms)

| ID | op | M x K x N | µs | bound | cores | DRAM % | FLOPs % | in0_bw | subblock | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 95 | rms_norm (time attn) | — | 721 | DRAM* | 110 | 82* | — | — | — | attention.py:64 |
| 96 | linear wg (time gate) | 98304x768x768 | 3327 | SLOW | 110 | 17.8 | 11.5 | 2 | 4x2 | attention.py:65 |
| 97 | SIGMOID (unfused activation of 96) | — | 1066 | SLOW* | 110 | 55* | — | — | — | attention.py:65 → matmul.cpp:380 |
| 98 | linear wqkv | 98304x768x2304 | 5853 | SLOW | 110 | 20.3 | 19.5 | 2 | 7x1 | attention.py:141 |
| 99 | split_qkv_and_split_heads | — | 2149 | DRAM* | 110 | 82* | — | — | — | attention.py:83 |
| 100 | rotary_embedding q | — | 1997 | SLOW* | 110 | 30* | — | — | — | attention.py:122 |
| 101 | rotary_embedding k | — | 1993 | SLOW* | 110 | 30* | — | — | — | attention.py:122 |
| 102 | SDPA causal | — | 1368 | DRAM* | 110 | 86* | — | — | — | attention.py:99 |
| 103 | concatenate_heads | — | 729 | DRAM* | 110 | 81* | — | — | — | attention.py:113 |
| 104 | MUL ctx*gate | — | 1000 | DRAM* | 110 | 88* | — | — | — | attention.py:73 |
| 105 | linear wo (time) | 98304x768x768 | 3354 | SLOW | 110 | 17.7 | 11.4 | 2 | 4x2 | attention.py:76 |
| 106 | ADD residual | — | 1002 | DRAM* | 110 | 88* | — | — | — | transformer.py:28 |
| 107 | rms_norm (variate attn) | — | 723 | DRAM* | 110 | 82* | — | — | — | attention.py:64 |
| 108 | linear wg (variate gate) | 98304x768x768 | 3414 | SLOW | 110 | 17.3 | 11.2 | 2 | 4x2 | attention.py:65 |
| 109 | SIGMOID (unfused activation of 108) | — | 1122 | SLOW* | 110 | 53* | — | — | — | attention.py:65 → matmul.cpp:380 |
| 110 | linear wv (univariate V path) | 98304x768x768 | 3282 | SLOW | 110 | 18.0 | 11.6 | 2 | 4x2 | attention.py:69 |
| 111 | MUL v*gate | — | 1008 | DRAM* | 110 | 88* | — | — | — | attention.py:73 |
| 112 | linear wo (variate) | 98304x768x768 | 3382 | SLOW | 110 | 17.5 | 11.3 | 2 | 4x2 | attention.py:76 |
| 113 | ADD residual | — | 1004 | DRAM* | 110 | 88* | — | — | — | transformer.py:28 |
| 114 | rms_norm (MLP) | — | 725 | DRAM* | 110 | 81* | — | — | — | mlp.py:31 |
| 115 | linear w_up | 98304x768x2048 | 5206 | SLOW | 110 | 20.9 | 19.5 | 2 | 2x4 | mlp.py:32 |
| 116 | SILU (unfused activation of 115) | — | 2602 | SLOW* | 110 | 60* | — | — | — | mlp.py:32 → matmul.cpp:380 |
| 117 | linear w_gate | 98304x768x2048 | 5169 | SLOW | 110 | 21.0 | 19.7 | 2 | 2x4 | mlp.py:33 |
| 118 | MUL gate*up | — | 2697 | DRAM* | 110 | 87* | — | — | — | mlp.py:35 |
| 119 | linear w_down | 98304x2048x768 | 7874 | SLOW | 110 | 13.8 | 12.9 | 2 | 4x2 | mlp.py:38 |
| 120 | ADD residual | — | 998 | DRAM* | 110 | 89* | — | — | — | transformer.py:29 |

Embed (rows 65–71, 22.1 ms):

- 65: hidden 98304x64x3072 + bias, 3727 µs. `in0_block_w=2`, subblock 2x4.
- 66: SILU, 2660 µs (residual_block.py:43).
- 67: output 98304x3072x768 + bias, **11748 µs**. This is the slowest single op (residual_block.py:44).
- 68: residual 98304x64x768, 961 µs (residual_block.py:46).
- 69: ADD (residual_block.py:47).
- 70: MUL keep (model.py:198).
- 71: ADD query (model.py:200).

Head (rows 123–130, 0.9 ms): slice (model.py:211), reshape (:213), rms_norm (:214), then 3 matmuls with `in0_block_w=1`, a SILU and an ADD.

## 6. Top 10 op roles (full model, with multiplicity)

The lower bound (LB) per call is max(FLOPs / 359 TFLOPs, bytes / 512 GB/s), where bytes = input + weight + output in bf16. When a matmul and its unfused activation are counted as one role, the LB is the fused matmul's LB. Headroom = (actual − LB) x n.

| # | role | n | full ms | per call µs | M x K x N | FLOPs % (304 / 359) | DRAM % | bound | LB/call µs | headroom ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MLP up 768x2048 + SILU | 24 | 192.4 (126.2 + 66.1) | 8015 (5259 + 2755) | 98304x768x2048 | 19.3 / 16.4 | 20.7 / 57* | SLOW | 1087 (DRAM) | **166.3** |
| 2 | MLP down 2048x768 | 24 | 191.6 | 7982 | 98304x2048x768 | 12.7 / 10.8 | 13.6 | SLOW | 1087 (DRAM) | **165.5** |
| 3 | QKV 768x2304 | 24 | 140.5 | 5853 | 98304x768x2304 | 19.5 / 16.6 | 20.3 | SLOW | 1187 (DRAM) | **112.0** |
| 4 | attn gate 768x768 + SIGMOID (24 time + 6 variate) | 30 | 132.2 (99.4 + 32.8) | 4405 (3312 + 1094) | 98304x768x768 | 11.5 / 9.8 | 17.9 / 54* | SLOW | 592 (DRAM) | **114.4** |
| 5 | MLP gate 768x2048 | 24 | 126.6 | 5273 | 98304x768x2048 | 19.3 / 16.3 | 20.6 | SLOW | 1087 (DRAM) | **100.5** |
| 6 | attn out wo 768x768 (24 + 6) | 30 | 98.0 | 3266 | 98304x768x768 | 11.7 / 9.9 | 18.1 | SLOW | 592 (DRAM) | **80.2** |
| 7 | rotary q + k | 48 | 95.9 | 1999 | eltwise (1024x12x96x64) | — | 29.5* | SLOW* | 590 (DRAM) | **67.6** |
| 8 | MLP MUL gate*up | 24 | 65.3 | 2722 | eltwise 98304x2048 | — | 86.7* | DRAM* | 2359 | 8.7 |
| 9 | residual ADD | 54 | 54.1 | 1003 | eltwise 98304x768 | — | 88.2* | DRAM* | 885 | 6.4 |
| 10 | create heads (split qkv) | 24 | 51.6 | 2148 | 98304x2304 → 3 x (1024,12,96,64) | — | 82.4* | DRAM* | 1769 | 9.1 |

Together these ten roles are 1148 ms (87.6 %). Next in line:

| role | n | full ms | headroom ms | notes |
|---|---|---|---|---|
| rms_norm | 54 | 39.0 | 7.2 | 82 %* DRAM |
| SDPA | 24 | 32.8 | 4.5 | 86 %* DRAM; math 3.9 % of HiFi2 |
| attn MUL | 30 | 29.9 | 3.4 | |
| variate wv | 6 | 19.7 | 16.1 | |
| concat heads | 24 | 17.5 | 3.4 | |
| embed out 3072x768 | 1 | 11.75 | 10.3 | |

Every large matmul has arithmetic intensity of 383–611 FLOP/B, below the HiFi2 ridge of 701 FLOP/B. With DRAM-interleaved activations, their roofline floor is therefore DRAM traffic, not math.

## 7. Findings, ranked by estimated full-model savings

**F1. The default matmul config is data-movement-bound and re-reads activations 3–6x.** Savings ≈ **600 ms** if every matmul reached 50 % of the HiFi2 peak (562 ms on the 110-core basis; 670 ms to the roofline). About **400 ms** comes from removing in0 re-reads alone. Confidence: high on the headroom, medium on the mechanism.

- Evidence rows: 65, 67, 68, 75, 77, 84, 87, 89, 91, 96, 98, 105, 108, 110, 112, 115, 117, 119. Matmuls are 818.6 ms (62.5 %), all SLOW at 11–20 % FLOPs and 13–21 % DRAM.
- Config: `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0=0`, so in1 (weights) is multicast and M is split 110 ways (`per_core_M=28`, i.e. 3072 tiles / 110). `in0_block_w=2` everywhere. `out_block_w < per_core_N`:

  | N | out_block_w / per_core_N | out_block_h |
  |---|---|---|
  | 768 | 8 / 24 | 28 |
  | 2048 | 16 / 64 | 14 |
  | 2304 | 18 / 72 | 14 |
  | 3072 (embed hidden) | 16 / 96 | 14 |

- The in0 reader loops `bh → bw → k-block` and re-reads its in0 rows from DRAM for every N-block (`reader_bmm_tile_layout_in0_sender_padding.cpp:259-265, 284-323`). Each block is 28–56 separate 2 KB tile reads followed by a barrier. Activations are therefore read **3x** (N=768), **4x** (N=2048/2304) or **6x** (64x3072).
- A one-parameter model, t ≈ 1.29 µs x (in0 tiles including re-reads + output tiles) per core, fits all 7 large shapes within −4…+13 %. That is ≈ 180 GB/s aggregate, 35 % of the DRAM peak:

  | shape | effective DRAM | MAC time at peak, % of measured |
  |---|---|---|
  | 768² | 184 GB/s | 12 % |
  | 768x2304 | 181 GB/s | 20 % |
  | 768x2048 | 192 GB/s | 19 % |
  | 2048x768 | 171 GB/s | 13 % |
  | 3072x768 | 168 GB/s | 13 % |
  | 64x3072 | 183 GB/s | 3 % |

- K-heavy shapes are slowest for their FLOPs (down 7982 µs vs up 5259 µs at equal FLOPs) because in0 is the larger operand and is re-read 3x.
- Removing re-reads at the same per-tile rate:

  | role | savings |
  |---|---|
  | down | 113.5 ms |
  | up + gate | 113.7 ms |
  | 768x768 x66 | 108.5 ms |
  | QKV | 60.2 ms |
  | embed out | 7.2 ms |
  | **total** | **403.5 ms** |

- The "50 % of HiFi2" target (217 ms of matmul) sits above the DRAM floor for every shape, so it does not exceed the roofline.
- tt-perf-report advice for every dominant matmul (verbatim): *"If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED) • in0_block_w=2 and output subblock 4x2 look good 🤷 • If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy"* (2x4 / 7x1 for the 2048/2304 shapes). The "look good" heuristic only flags `in0_block_w=1`, so it misses the re-read problem. Placing the whole input 0 in L1 is infeasible: 151–403 MB against ≈165 MB of total L1 on 110 cores. It only works with M-chunking.
- The CB-wait columns are empty, so reader stall versus packer stall can't be separated. The mechanism rests on the fitted model.

**F2. Padding 66 → 96 tokens wastes 31.25 % of per-token work.** Upper bound **409 ms** (0.3125 x 1309.1 ms of non-head work); realistic net ≈ **270–295 ms**. Confidence: medium on the waste, low on the net.

- Evidence: every per-token op runs on M = 98304 = 1024 x 96.
- 1024 x 66 = 67584 = **2112 tiles is itself tile-aligned**. The flat per-token ops (linears, norms, eltwise: 1111 ms) don't need per-row padding; only the (rows, S_pad, D) attention view does (create heads, rotary, SDPA, concat: 197.8 ms).
- Matmul time follows `per_core_M`: 28 → 20 tiles, i.e. −28.6 %.
- Gross recoverable with the attention core kept padded: 0.286 x 818 + 0.3125 x (490.8 − 197.8) = 234 + 92 = **325 ms**.
- Cost: one padded↔packed re-layout pair per time-attention sublayer. The 66-row series boundaries fall mid-tile, so this is row-granular data movement of ≥ 255–764 MB per conversion, about ≥ 0.6–1.7 ms each, ≈ 30–55 ms over 24 sublayers.
- Overlaps multiplicatively with F1, F3 and F4.

**F3. Unfused activations: 101.7 ms (7.8 %) in 56 ops.** Savings ≈ **90–100 ms** and −56 ops (11 % of ops). Confidence: high.

- Breakdown: SILU 68.9 ms (MLP 66.1 ms in 24 ops, embed 2.66, head 0.10); SIGMOID 32.8 ms (30 ops).
- Evidence: every matmul followed by an activation (65, 75, 87, 96, 108, 115, 126) has `fused_activation=std::nullopt` and `user_fused_activation=std::nullopt`, followed by a `UnaryDeviceOperation` with `op_type=SILU`/`SIGMOID` (rows 66, 76, 88, 97, 109, 116, 127).
- Root cause in `ttnn/cpp/ttnn/operations/matmul/matmul.cpp:323-330`: when `ttnn.linear` gets `activation=` but no `core_grid` (`user_core_coord`), the activation is stripped from the kernel and applied as a trailing `unary_chain` (`:380-383`).
- The standalone unaries also run at only 53–60 %* of DRAM peak, versus 89 %* for the embed SILU. They are a full extra DRAM round trip: 2 x 151 MB for sigmoid, 2 x 403 MB for SILU.
- Savings = 101.7 ms minus the in-kernel SFPU cost, which should be small because the matmuls are not math-bound.
- Alternative absorption point: the following MUL (BinaryNg `lhs_activations`/`rhs_activations`, currently `{}`).

**F4. Rotary embedding: 95.9 ms (7.3 %) in 48 calls of 2.0 ms each, 29.5 %* of DRAM peak.** Savings ≈ **63 ms** if it ran at 85 % of DRAM peak on 2 x 151 MB ((1999 − 694) µs x 48), up to 95.9 ms if it were eliminated or fused. Confidence: high on the inefficiency, medium on what is attainable.

- Evidence rows: 79, 80, 100, 101.
- The prefill reader (`reader_rotary_embedding_interleaved_start_id.cpp:85-124`) issues **4 serialized single-tile reads per output tile** (rotated-half input, sin, input, cos), each followed by `noc.async_read_barrier()`. The input is read twice, and the broadcast (1,1,96,64) cos/sin tables are re-read for every one of the 12,288 (row, head) slices.
- Physical traffic ≈ 5 x 151 MB = 755 MB/call, ≈ 74 % of peak and latency-bound. Compute per tile: 3 muls + 1 add (+ a −1 scalar mul on half the tiles) through intermediate CBs.
- Partial RoPE is implemented as zero `inv_freq` past rope_dim/2 (weights.py:155). Those dims get an identity rotation but are still processed.

**F5. TM ops, create heads + concat heads: 69.1 ms (5.3 %).** Savings: up to **69 ms**, only by eliminating the passes; per-op tuning is worth just 12.5 ms. Confidence: medium.

- Evidence rows: 78, 82, 99, 103.
- They already run at 81–82 %* of DRAM peak. Create heads moves 906 MB (453 in, 453 out) in 2.15 ms; concat heads 302 MB in 0.73 ms.
- Removing them requires layouts that the QKV matmul, SDPA or wo can consume directly.

**F6. Binary ops: 152.4 ms (11.6 %) in 112 ops, all at 86–89 %* of DRAM peak.** Savings beyond F3: ≈ **1–17 ms**. Confidence: high.

- ADD: 56.2 ms in 57 ops (54 residual adds = 54.1 ms, 2 in embed, 1 in head).
- MUL: 96.2 ms in 55 ops (MLP gate*up 65.3 ms in 24, attention ctx*gate 29.9 ms in 30, embed keep 1.0 ms).
- Per-op headroom is only ≈ 17 ms.
- Fusable:
  - The 54 attention/MLP MULs can absorb the preceding activation (counted in F3).
  - Embed keep-MUL + query-ADD (rows 70, 71: 2.0 ms) together just swap in 2 decoder rows and could be ≤ 1 op.
- Not fusable with today's ttnn ops: the residual ADDs. `ttnn.addmm` is matmul + a separate `add_` (matmul.cpp:607-620), and `rms_norm(residual_input_tensor=)` does not return the sum.

**F7. Shared-input linears are not merged** (wg + wqkv, w_up + w_gate, variate wg + wv). Savings ≈ **16 ms**, and only after F1. Confidence: low.

- This means 54 extra reads of `xn` (151 MB, ≈ 0.3 ms each at peak) and 54 extra ops.
- Under the current re-reading kernel, merging N does not cut traffic because re-reads scale with N-blocks. It only pays off once F1 is fixed.

**F8. Other ops near their floor:** rms_norm 39.0 ms (82 %*) and SDPA 32.8 ms (86 %*; causal over 6 of 9 chunks; math is 3.9 % of HiFi2) have ≈ **12 ms** of headroom combined.

**Structural ceiling (applies to all findings).**

- The current graph streams **≈ 249 GB per forward through DRAM**: matmuls 76 GB, all other ops 172 GB. The DRAM floor is therefore **485 ms** (428 ms with activations fused), even if every op hit its roofline.
- Once F1 is fixed, the non-matmul ops dominate the floor. Going below it needs fewer DRAM round trips (fusion, L1-resident M-chunks) or fewer tokens (F2).

**Host side (wall-clock only, not device).**

- Fixed prepare + upload + postprocess is 106.7 ms per call (7.4 % of wall). Up to ≈ 100 ms of wall time is addressable. It was not profiled at function level this round, so confidence is low.
- Tracing saves ≈ 0 device ms today. tt-perf-report's "Running with tracing could save 2599 μs" comes from the sync-inflated boundary gaps and the head ops issued right after a sync.
- Host issue (≈ 150–200 µs/op under Tracy, ≈ 80–100 ms per forward) becomes exposed only if device time falls below ≈ 100 ms.

**Requested quantities**

| | full-model ms | detail |
|---|---|---|
| (a) unfused unary activations | **101.7** | 56 ops, 7.8 % |
| (b) binary ops | **152.4** | 112 ops. Fusable: 54 MULs can absorb activations (F3); embed MUL+ADD 2.0 ms. Not fusable today: residual ADD 54.1 ms. Per-op headroom ≈ 17 ms |
| (c) padding waste | **409** | upper bound; 325 gross with the attention core padded; ≈ 270–295 net |
| (d) matmul headroom at 50 % of HiFi2 peak | **601** | 359 basis; **562** on the 110-core 304 basis; 670 to roofline; 404 from removing re-reads alone |
| (e) rotary + create heads + concat heads | **165.0** | 95.9 + 51.6 + 17.5 = 12.6 %; headroom to floor 67.6 + 9.1 + 3.4 = 80.1 ms |

Savings overlap and are not additive. For example, F2 scales down whatever F1, F3 and F4 leave behind.

## 8. Comparison with the performance precision (bf8 weights, LoFi)

- Device: 1307.2 → 1251.3 ms (**−55.9 ms, −4.3 %**). Wall: 1445.3 → 1353.0 ms (−92.3 ms, −6.4 %). Of the wall difference, 12.0 ms is host prepare/upload/postprocess noise (the host code is identical) and ≈ 24 ms is the bimodal run-to-run variance of the accuracy runs.
- bf8 weights only halve weight bytes: ≈ 0.4 GB of weights per forward against ≈ 249 GB of activation traffic, so they make no difference. LoFi halves the math phases compared with HiFi2.
- If the matmuls were math-bound, LoFi would save up to ≈ 409 ms (½ x 818.6). It saved 56 ms, i.e. **6.8 % of matmul time**.
- The FLOPs-utilization data predicts this: exposed HiFi2 math ≈ 15.6 % x 818.6 = 128 ms, so halving it saves ≈ 64 ms. The observed 56 ms matches.
- Implication: the matmuls are data-movement-bound, and math is only ≈ 1/7 of their time. The 491 ms of non-matmul device time (37.5 %) is unaffected by precision.
- Fidelity and weight dtype are not the lever; activation traffic is (F1 re-reads, F2 padding, F3 extra round trips, F4 read amplification).
- A performance-precision profile was not captured, so this split is inferred from wall/device deltas.
