# Round 1 proposals (from round 0, `iter_00`; mapped to the optimize skill)

This file is the round's analysis record (PLAN section 9: no `analysis.md`), together with `summary.md`,
`summary.json` and `perf_report.*`. The analysis was done by a read-only analyst subagent over `iter_00/` and the
code; I checked every headline number below against `summary.json` / `bench.json` (the analyst's numbers matched).

## Headline (round 0)

- **Target** `forward` paper (1024 series x 2048 -> 64; 285 tokens padded to 320) / accuracy, chip 1, traced
  replay, realistic input mix (bench job 499): **device 3856.9 ms**, **steady wall 4518.4 ms**; CPU reference
  413.4 s (8 threads): 91x. The profile and the analysis below used the first (smooth) bench inputs: device
  3855.3 ms (5 runs 3853.0-3858.7), wall 4428.6 ms (job 440); device time does not depend on the values. Gate
  metrics: see GATE.md.
- **Reconciliation:** reduced-layer estimate **3835.0 ms vs 3856.9 ms measured (-0.6 %)** (profile job 455,
  bench job 499; -0.5 % against job 440).
  The estimate sums kernel durations; the traced replay adds device-side op-to-op gaps. 20-22 ms / 47070 ops =
  0.43-0.47 us per op, matching the 0.43-0.56 us per op measured in the device-bound sections of the eager profile.
  The eager `Op-to-Op Gap` of the `slstm_steps` rows (2974 ms) is host dispatch plus profiler drains and does
  not exist in the trace. sLSTM steps are scaled from 71 of 284 profiled steps (x24): per-step kernel time
  546.5 us, sigma 1.4 us over the 71 steps.
- **Where the time goes (full model, computed from `sections/*.csv` x counts):**

  | part | ms | share |
  |---|---|---|
  | sLSTM time mixers (6) | 1646.4 | 42.9 % |
  | - pre (norm, transpose to time-major, `W_in`, fp32 bias add) | 326.9 | |
  | - recurrence (1704 steps x 546.5 us, 27 ops each) | 932.1 | |
  | - post (concat, per-head norm, transpose back, residual) | 387.4 | |
  | mLSTM time mixers (6) | 1252.2 | 32.7 % |
  | - of which the chunkwise core (116 ops) | 644.5 | |
  | FFNs (12) | 782.9 | 20.4 % |
  | embed + head | 153.5 | 4.0 % |

- **Category split:** matmuls 1673.6 ms (43.6 %), elementwise and norms 1554.7 ms (40.5 %), data movement
  (slice, permute, concat, head split/concat, (un)tilize, typecast, transpose) 606.9 ms (15.8 %). Matmul FLOPs
  utilization: **9.7 %** (`summary.json` `matmul_weighted_flops_percent`, the number of record); 8.6 % FLOP-weighted
  against the 110-core peak. 34.7 TFLOP per forward = 149 ms at peak; matmul bytes 164 GB = 320 ms at 512 GB/s.
  The non-matmul ops run at about 72 % of the DRAM floor of the bytes they move (1549 of 2161 ms): **the cost is
  the graph (rows, bytes, op count), not slow kernels.**
- **Structural waste:** 192 of 320 time rows are constant left pads (60 %) and 35 more are chunk padding (11 %);
  the dense matmuls run as 1024 batched 10-tile problems (`b={1024} x 320 x K`) with `in0_block_w` 1; each sLSTM
  step makes 27 DRAM round trips (165 MB moved to carry about 6 MB of state); only tokens 255-256 of the head
  output are read.

## Op-to-op map per layer kind (one instance; ms; [P] padding, [R] redundant or fusable, [L] layout move)

- **embed (66.4):** 2108 hidden 64->2048+bias 7.55 (ib1) | 2109 SiLU 6.35 [R] | **2110 out 2048->512 46.59**
  (80 cores, ib1, 1x2, 6.7 % FLOPs) | 2111 skip 3.64 | 2112 add 2.23. All 320 rows, 71 % [P].
- **mLSTM block (274.1, x6, 142 ops):** RMSNorm 1.69 | 2116 QKV 512->1536 19.00 (ib1, 1x1; 512 zero columns from
  the q/k 64->128 pad [P]) | heads + q/k slices 6.88 [L] | **2120-2123 four gate matmuls 512->32, fp32 weights,
  10 cores: 31.05** [R] | 2124 o-proj 12.96 | gates concat/tanh/x15 2.57 | **chunkwise core 2128-2243: 107.42**
  (f column via untilize/row-major slice/tilize 10.10 [L]; cumsum via permute/accumulate/permute 5.36 [L]; D build
  8.09; row max + g_last via untilize/slice/tilize 5.15 [L]; state carry 4 chunks x 16 ops on 32x-padded scalar
  tiles 18.48 [P]; W = exp(D-m)*S 10.19; bmms 2153, 2222, 2230, 2231, 2234 DRAM-bound at 78-82 %; numerator,
  denominator, divide 21.32) | LN per head 3.65 | concat heads 3.20 | sigmoid(o)*h 5.13 [R] | 2248 out-proj 12.95 |
  add 2.21 | **FFN 65.39** (gate 16.96, SiLU 6.76 [R], up 16.97, mul 6.61 [R], down 14.18).
- **sLSTM pre (54.49, x6):** RMSNorm 1.69 | **2260 permute to time-major 15.60** [L] (43 GB/s) | 2261 `W_in`
  512->2048 HiFi4 fp32-out 19.47 (ib1, 2x2; 3/4 of the FLOPs on zero blocks [P]) | **2262 fp32 bias add 17.74** [R]
  (5.4 GB moved).
- **sLSTM step (546.5 us, x1704):** slice Wx_t 42.6 | y_bf16 @ R 98.2 (88 cores, ib1, 14 % of chip FLOPs) |
  ry + wx 59.1 | 4 gate slices 48.6 [L] | tanh, sigmoid 24.1 | log-sigmoid (max, sigmoid, log) 35.4 | lfm, m'
  35.1 | two sub+exp 59.1 | c update 52.6 | n update 46.4 | y = o c / n 36.2 | typecast 9.2. All DRAM interleaved
  fp32 [1024, 512|2048]; non-matmul DRAM floor 321.8 us.
- **sLSTM post (129.66, x6):** concat 285 y + zero tail 6.54 [L][P] | **per-head LN 47.49** (two fp32 512x512
  block-averaging matmuls 4198/4201 at 11.71 each, ib1, plus 6 elementwise 24.07) | 4206 permute back 8.32 [L] |
  add 2.21 | FFN 65.09.
- **head (87.13, x1):** T5 norm 1.69 | 4218 hidden 20.44 | SiLU 6.04 [R] | **4220 out 2048->288 45.76** (ib1, 1x1,
  3.4 % FLOPs) | 4221 skip 11.74 | add 1.26 | window slice 0.21. 320 tokens computed, 2 read [P 99 %].

## Top-10 op roles (full model)

| # | role | M / K / N | config | ms x count = total | FLOPs % (used cores / 110) | fidelity, dtypes |
|---|---|---|---|---|---|---|
| 1 | FFN gate, up (2251, 2253, 4209, 4211) | b1024x320 / 512 / 1408 | default 2D, ib1, 1x4 | 16.96 x 24 = 407.2 | 9.2 / 9.2 | HiFi2 bf16 |
| 2 | sLSTM step BinaryNg (11 per step) | [1024, 512] | DRAM interleaved | 0.0176 x 18744 = 330.7 | - | fp32 |
| 3 | mLSTM gate projections (2120-2123) | b1024x320 / 512 / 32 | 1D, **10 cores**, ib2, 1x1 | 7.76 x 24 = 186.3 | 10.0 / 0.9 | HiFi4 bf16 x fp32 |
| 4 | FFN down (2255, 4213) | b1024x320 / 1408 / 512 | 80 cores, ib4, 1x2 | 14.21 x 12 = 170.5 | 15.0 / 10.9 | HiFi2 bf16 |
| 5 | sLSTM step R matmul | 1024 / 512 / 2048 | 2D, 88 cores, ib1, 2x2 | 0.0982 x 1704 = 167.4 | 18.0 / 14.4 | HiFi4 bf16 -> fp32 |
| 6 | sLSTM step Unary (8 per step) | [1024, 512] | DRAM interleaved | 0.0119 x 13632 = 161.6 | - | fp32 |
| 7 | mLSTM o-proj, out-proj (2124, 2248) | b1024x320 / 512 / 512 | 80 cores, ib1, 1x2 | 12.96 x 12 = 155.5 | 6.0 / 4.4 | HiFi2 bf16 |
| 8 | sLSTM post per-head-LN elementwise | [320, 1, 1024, 512] | DRAM interleaved | 4.01 x 36 = 144.5 | - | fp32 -> bf16 |
| 9 | sLSTM post per-head-LN matmuls (4198, 4201) | b320x1024 / 512 / 512 | 64 cores, ib1, 2x2 | 11.71 x 12 = 140.5 | 16.6 / 9.7 | HiFi4 fp32 |
| 10 | sLSTM `W_in` (2261) | b320x1024 / 512 / 2048 | 88 cores, ib1, 2x2 | 19.47 x 6 = 116.8 | 29.0 / 23.2 | HiFi4 bf16 -> fp32 |

Next: QKV 114.0, pre bias add 106.4, step ry + wx 100.7, pre permute 93.6, step gate slices 82.8, FFN multiply
79.9, FFN SiLU 78.4. Matmuls with `in0_block_w` 1 or 1x1 subblocks that are a material share: 2110, 2116,
2120-2123, 2124/2248, 2251/2253/4209/4211, 2261, the step matmul, 4198/4201, 4218, 4220/4221 (all ttnn defaults;
nothing tried yet).

**Why the dense matmuls are slow** (analyst, from `matmul_program_config.cpp`): in0 is `[1024, 1, 320, K]` with
an in1 of batch 1, so ttnn picks a 2D multicast config on M = 320 (10 tiles) with the batch not fused:
`per_core_M` 1 forces subblock height 1, `in0_block_w` is `Kt/11` only when 11 divides `Kt` (so 1 for K = 64, 512,
2048), and the result is 1024 batch iterations of 1 x Nb blocks. Without a program config, `activation="silu"` is
applied as a trailing unary. Flattening to `[1, 1, rows*seq, K]` (a free view: 320 is a tile multiple) gets the 1D
default (~2x, by Timestral's round 0 at 98304 rows); explicit 2D configs or `minimal_matmul` reached 44-60 % in
Timestral, here DRAM-bound at ~60 % (~4x).

**Guards.** paper/performance 3792.3 ms (-1.6 %: LoFi/bfp8 barely move config-bound matmuls; probably mostly
`W_in` going HiFi4 -> LoFi). long_context 370.6 ms: same 47070 ops on 32 rows (7.9 us per op), so per-op latency
dominates and only the fused sLSTM kernel moves it much; it has no pads, so R1-1 does not apply and every new
config must be shape-keyed. multivariate 10990.8 ms = target + 12 x 594.6 ms per variate block: its dense matmuls
run as `b={10240} x 32 x 512` (one tile of M per batch), plus two [1024, 1, 320, 512] transposes per block; R1-2's
flattening should help it most. Profile `MODEL_PROFILE_SHAPE=multivariate` and `long_context` in round 1 before
quoting guard savings.

## Proposals

Savings overlap and are not additive; "after R1-1" means once the constant pad is gone. Gates for every item: the
PCC suite (all tests in one `pcc` job), target `pcc_normalized` within 0.0005 of round 0 and every floor holding,
target device time >= 2 % better than a **back-to-back baseline re-run on chip 1**, no guard regressing > 3 %.

| # | proposal | finding | est. full-model saving | confidence | risk / gate |
|---|---|---|---|---|---|
| R1-1 | constant left-pad skip (candidate b): cached per-layer states after the 192 pad tokens, run tokens 192..284 | F1 | 1.84-2.66 s (-48..-69 %) | high (mechanism exact on CPU), medium (ms) | medium: two paths, PCC tests that assert the path |
| R1-2 | dense matmul topology and configs: flatten M, fused activations, one gate matmul on 110 cores, `W_in` bias in the matmul, then a per-role config sweep | F2, F7 | 1.18-1.35 s at T = 320; 0.35-1.1 s after R1-1 | medium-high | low-medium: per-shape configs, sweep table per role |
| R1-3 | read-out cut (candidate d): head and block 11 after its recurrence on the read-out window only | F8 | 0.19 s at T = 320; 37-59 ms after R1-1 | high (exact) | low |
| R1-4 | mLSTM chunkwise stock fixes: f in its own tile column, cumsum as one triangular bmm, g_last as a chunk sum | F4 (part) | 0.11 s; 32 ms after R1-1 | high | low |
| R1-5 | sLSTM pre/post layout: per-head norm in fewer ops, cheaper transposes | F6 (i), F7 | 0.18-0.30 s at T = 320; ~0.07 s after R1-1 | medium | low; micro-benchmark the transposes |
| R1-6 | fused sLSTM time-loop kernel (candidate a), model-local `generic_op` | F3, F6 (ii) | 0.80-0.90 s at T = 320; 0.26-0.29 s after R1-1 (+ up to 0.1 s of post) | medium | high: **micro-benchmark first**, gate <= 40 us/step at 1024 rows, PCC >= 0.9999 vs the op-level loop |
| R1-7 | host: vectorized prepare (348 ms) and postprocess (193 ms) | wall | wall only, ~0.3-0.4 s | medium | low; judged on wall and the host split |
| R1-8 | precision: embed (patch encoder) at HiFi4 | PLAN 9 | accuracy only (cost ~+10-20 ms) | measured | low; see GATE.md |
| - | not proposed now: F5 (sLSTM step in L1 with stock ops: subsumed by R1-6), F9 (stop non-last sLSTM blocks at t = 256: needs a contract change, ~5 ms after R1-6), F10 (sign-flip batching: stage E) | | | | |

### R1-1: constant left-pad skip (pre-registered candidate b)
- **Hypothesis / mechanism.** At context <= 8192 - 32 P, the first P tokens are all-NaN patches whose token is
  `[0 x 32 | 0 x 32]`; every layer's states after them are input-independent (stage A: bitwise equal across 4
  inputs, `pad_check.json`). Compute them once per P on the host with the fp32 dataflow (one all-pad row; also
  more accurate than the device), upload them as per-shape constants, and run only tokens P..284 (93 tokens at
  context 2048): mLSTM starts its chunk carry from (C, n, m), sLSTM from (y, c, n, m).
- **Evidence.** Per-token sections 2902.8 ms at T = 320 (embed 66.4, mLSTM 1644.5, pre 326.9, post 777.9, head
  87.1); steps 931.2 ms for 284 ordinary steps. CPU prototype from cached states reproduces the upstream head
  output to 8.7e-7 relative with chunk 32, 64 or 96 (`tests/golden/sim_padcut.py`).
- **Estimate.** Steps 931.2 x (1 - 92/284) = 629.5 ms saved. Per-token parts at T' = 96: 0.7 x 2902.8 = 2032 ms,
  total 2.66 s; but the current batch-major defaults barely shrink with M (1169.1 ms of dense matmuls), so without
  R1-2: 0.7 x (2902.8 - 1169.1) + 629.5 = 1.84 s. T' = 128 (chunk 64) saves 0.6 instead of 0.7 of the per-token
  part (-290 ms); T' = 96 needs chunk 32 (3 chunks) or 96 (1 chunk), exact on CPU (2e-5 relative in fp32).
- **Rules.** "Optimize the operation topology before fine-tuning individual op configs"; padding is waste.
- **Plan.** `model.py`: detect P (leading all-NaN tokens common to every row); `reference/dataflow.py`: per-layer
  states for P (and for grouped rows when the variate mixer runs); `layers.py`: initial states for the chunkwise
  mLSTM and the sLSTM loop; window and read-out offsets by P. Keep the full path for P < 64 (long_context).
- **Experiment and gate.** CPU check of the grouped-row pad states (multivariate) first; PCC tests that force and
  assert each path (model16 and paper rows: cut; long_context: full; multivar4: cut with groups); then bench.
- **Risks.** Mixed-length batches (P is the minimum over rows); multivariate pad states depend on whether a row
  is in a group (fallback: full path for multivariate).

### R1-2: dense matmul topology and program configs
- **Mechanism.** (a) Reshape activations to `[1, 1, rows*seq, K]` around every dense linear (a view) so the batch
  folds into M; (b) fuse SiLU (and the SwiGLU product via `minimal_matmul(fuse_swiglu=True)` as in Timestral) and
  the biases into the matmuls; (c) one mLSTM gate matmul 512 -> 2x4 on 110 cores instead of four 10-core ones
  (keep fp32 weights; regular matmul config); (d) the `W_in` fp32 bias in the matmul (removes 2262: 17.74 ms x 6);
  (e) pack QKV with o and drop the q/k zero padding where the head split allows; (f) a per-role sweep of explicit
  2D configs at the post-R1-1 M (98304 or 131072 rows), with a table per role.
- **Evidence and estimate (analyst F2, target = max(bytes / 0.6 x 512 GB/s, FLOPs / 0.45 peak)).**

  | role (calls) | now ms | target per call | saves ms |
  |---|---|---|---|
  | FFN gate + up + SiLU + mul -> fused SwiGLU (12) | 565.5 | 6.90 | 483 |
  | FFN down (12) | 170.5 | 4.10 | 121 |
  | QKV + o packed (6) | 191.8 | 5.46 | 159 |
  | mLSTM gate projections on 110 cores (24 -> 6) | 186.3 | 1.23 x 4 | 157 |
  | out-proj (6) | 77.8 | 2.19 | 65 |
  | sLSTM post LN matmuls (12), if kept | 140.5 | 4.37 | 88 |
  | `W_in` + bias in the matmul (6) | 116.8 + 106.4 | 10.04 | 57 + ~100 |
  | embed, head | 148.1 | 23.4 | 125 |
  | **total** | **1703.7** | | **~1355** (1.18 s at 40 % DRAM) |

  After R1-1 the targets scale by ~0.3: saving 0.35 s if today's configs scaled with T, up to ~1.1 s if not.
- **Rules.** "Prefill activations ... use 2D matmul program configs for large prefill matmuls"; "Explicitly
  configure memory_config, program_config, and compute_kernel_config"; Matmul Choices (packed projections,
  in0_block_w >= 2 is a floor, subblocks >= 2, leave a table per dominant role).
- **Experiment and gate.** Sub-items (a)+(b) first (no explicit configs), then (c), (d), (e), each measured; the
  config sweep (f) in its own script at the shapes the model runs, with PCC per candidate; shape-keyed configs so
  long_context (M = 10240) and multivariate stay within 3 %.

### R1-3: read-out cut (pre-registered candidate d)
- **Mechanism.** After block 11's recurrence everything is per token: run block 11's concat/norm/residual/FFN,
  the final norm and the head on the tile window holding the read-out tokens only; block 11's recurrence can stop
  at the last read-out token. Exact (Timestral R3-3: bit-identical).
- **Evidence and estimate.** head 87.13 ms and block-11 post 129.66 ms both on 320 rows for 2 read rows: 0.8 x
  87.1 + 0.8 x 129.7 + 28 x 0.55 = ~189 ms at T = 320; 37-59 ms after R1-1 (tokens 63-64 of a 96-row window
  straddle two tiles; 59 ms if gathered into one).
- **Experiment and gate.** PCC suite (H = 64 and H = 928 windows); bench.

### R1-4: mLSTM chunkwise stock-op fixes
- **Evidence.** 2129-2131 untilize/row-major slice/tilize for the f column (10.10 ms); 2134-2136 permute/
  accumulation/permute for the cumsum (5.36 ms); 2143-2145 untilize/slice/tilize for g_last (4.05 ms).
- **Mechanism.** Emit i and f in separate tile columns (f at column 32 of a 64-wide gate output); compute
  `b_row = lf_row @ U` with the upper-triangular ones matrix as one bmm (also removes transpose 2137); g_last as
  the chunk sum of `lf`. ~18 ms per layer = ~108 ms (~32 ms after R1-1).
- **Rule.** "Avoid data movement before tuning math."

### R1-5: sLSTM pre/post layout
- **Evidence.** 2260 permute 15.60 ms and 4206 permute 8.32 ms at 8-16 % of DRAM; per-head LN 47.49 ms per block
  (4198/4201 + 6 elementwise) vs 3.65 ms for the mLSTM's `ttnn.layer_norm` on the same element count.
- **Mechanism.** Per-head norm with `ttnn.layer_norm` on a head-split view (or fused elementwise), the zero tail
  dropped from the concat, and a micro-benchmark of cheaper transpose forms (CN then HC, row-major).
- **Estimate.** 0.18-0.30 s at T = 320, ~0.07 s after R1-1. Subsumed by R1-6 (ii) if the kernel writes normalized
  bf16 y directly.

### R1-6: fused sLSTM time-loop kernel (pre-registered candidate a)
- **Evidence.** 1704 steps x 546.5 us = 931.2 ms; 27 ops and 165 MB of DRAM traffic per step for ~6 MB of state;
  the step matmul runs at 14 % of chip FLOPs.
- **Mechanism.** A model-local `ttnn.generic_op` (absolute kernel paths, Chronos-2's descriptor pattern;
  `chunk_gated_delta_rule` as the program-structure template): work units (row tile, head) = 32 x 4 = 128 for 1024
  rows; each core keeps R_h (128 KB bf16) and its c, n, m, y tiles in L1 and loops over t; per step it reads only
  Wx_t (8.4 MB in total, ~16 us at peak) and writes y_t. Compute per step: 0.54 GFLOP of block-diagonal matmul
  (3.5 us at the HiFi4 peak) plus ~20 SFPU ops on ~5 tiles per core (~35 us).
- **Estimate.** 30-80 us per step: 0.80-0.90 s at T = 320, 0.26-0.29 s after R1-1; plus 0.32-0.37 s (0.1 s after
  R1-1) if it also emits the per-head-normalized bf16 y.
- **Experiment and gate.** Micro-benchmark first at the paper shape: **<= 40 us per step at 1024 rows (<= 11.4 ms
  per layer at 285 steps, <= 3.7 ms at 93)**, PCC >= 0.9999 against the op-level loop on the same inputs and
  32-step states >= 0.9999 against the goldens. Then the model, the suite, the bench.
- **Risks.** New kernel (L1 budget, multicast of y across heads is not needed thanks to the block-diagonal R, but
  row tiles of 32 x 4 heads = 128 units need 2 per core on 110 cores); the op-level loop stays as the fallback and
  the reference for the micro-benchmark.

### R1-7: host prepare and postprocess (wall only)
- **Evidence.** bench split medians: prepare 348 ms, upload 89 ms, postprocess 193 ms (wall - device = 573 ms).
- **Mechanism.** Vectorized scaler/tokenize without the 9120-step NaN padding copy; postprocess on the read-out
  window only, in one tensor op (Timestral R3-5: -59 ms wall).
- **Gate.** Wall and host split only, device unchanged.

### R1-8: patch encoder at HiFi4 (precision, accuracy policy only)
- **Evidence (job 469, long_context, eager):** embed at HiFi4: `pcc_row_min` 0.99485 -> 0.99591, median error
  6.3e-2 -> 4.5e-2 of the scale; every dense matmul at HiFi4: 0.99808, 4.2e-2. Smaller than t0's effect; worth it
  only if the G1 median or row-PCC floors need it.
