# Round 1 proposals (from the round-0 profile `iter_00/`, mapped to the optimize skill)

This file is the round's analysis record with `summary.md`, `summary.json` and `perf_report.*` (PLAN section 9 v1.1:
no `analysis.md`). The read-only analyst ran on the first round-0 profile, taken before the v1.1 policy change (bf16
dest accumulation; kept in `pre_v1.1/`). Its mechanisms and row reading hold; every number below is re-derived from
the final round-0 profile (ttq job 526) and bench (jobs 554) by the worker, with the arithmetic shown.

## Preamble

### Headline (target `forward`, `paper`, `accuracy`: 2.5B-FT, 1024 x 2048 -> 64, realistic-mix inputs)

| | value |
|---|---|
| precision | bf16 weights / activations; HiFi2 matmuls and SDPA with fp32 dest accumulation; patch embedding HiFi4 + fp32 acc; head fp32 output; RMSNorm HiFi4 + fp32; RoPE HiFi4 (`tt/common.py` table) |
| device (median of 5, 0.3 s idle) | **8209 ms** |
| steady wall (median of 5) | **8412 ms** |
| host split | prepare 85 ms, upload 10 ms, postprocess 13 ms |
| reduced-window estimate | **7984.0 ms** over 874 ops (`summary.json`) |
| gate metrics | pcc_normalized 0.99993, pcc_spread 0.99990, pcc_row_p01_active 0.9992, spread_ratio p01/p99 0.975/1.044, pcc_final / pcc_final_spread 1.0000 (job 554, no floor missed) |
| shape | 66 tokens per row (64 context + 2 masked) padded to S_pad = 96: every per-token op runs on M = 98304 rows, of which 67584 are real (68.75 %) and 66560 live (token 65 is never read) |

### Reconciliation

The reduced window (time layer 0, the last time layer, the variate layer, embed and head, scaled by `layer_counts.json`) gives 7984.0 ms against 8209 ms measured: -2.7 %. The profile is taken op by op with a device sync at each section signpost, so it omits the op-to-op dispatch gaps inside the 46 repeated layers (about 4.9 ms per layer, 874 ops); within the skill's 3 % bound, above PLAN G1 item 4's 2 %, explained here.

| section | device ms | ops | x count | ms |
|---|---|---|---|---|
| embed | 49.43 | 5 | 1 | 49.4 |
| layer_time | 170.60 | 18 | 46 | 7847.7 |
| last_time | 83.52 | 22 | 1 | 83.5 |
| layer_variate | 2.12 | 13 | 1 | 2.1 |
| head | 1.25 | 6 | 1 | 1.2 |
| **total** | | 874 | | **7984.0** |

### Category split and matmul utilization

- Matmul 5635.6 ms (70.6 %, 246 ops); non-matmul 2348.4 ms (29.4 %, 628 ops). Compute / TM / Other 93.95 / 5.44 / 0.62 %
  of the reduced window.
- **`matmul_weighted_flops_percent` = 31.29 %** (`summary.json`, the number of record). All matmuls do 462.9 TFLOP per
  forward: 1523 ms at the 110-core HiFi2 peak of ~304 TFLOPs (1289 ms at 359) against 5636 ms measured. Their
  intensity (1013-1612 FLOP/B) is above the 594 FLOP/B ridge, so their floor is math.
- The non-matmul ops move ~818 GB per forward (analyst, unchanged by the policy): a 1598 ms DRAM floor against 2348 ms;
  most run at 80-88 % of DRAM peak, so only fewer bytes (fusion, fewer tokens) cut them. RoPE (36 %) and the
  5472-wide SiLU (53 %) are the exceptions.

| op type (full model, sections x counts) | count | ms | % |
|---|---|---|---|
| Matmul | 246 | 5635.6 | 70.6 |
| BinaryNg (mul g*silu(u), residual mul beta*x, residual add, embed/head add) | 242 | 792.0 | 9.9 |
| RotaryEmbeddingLlama (q, k) | 94 | 406.6 | 5.1 |
| Unary (the SiLU that `ttnn.linear(activation=)` runs as a trailing op with the auto config) | 50 | 400.7 | 5.0 |
| NlpCreateHeads | 47 | 270.4 | 3.4 |
| SDPA | 47 | 204.0 | 2.6 |
| LayerNorm (RMSNorm) | 97 | 179.3 | 2.2 |
| NLPConcatHeads | 47 | 92.3 | 1.2 |
| untilize / slice / tilize / reshape (last-layer cut) | 4 | 3.2 | 0.0 |

### Op-to-op map (ms per op; IDs are `ops_perf_results.csv` lines)

- **embed** (IDs 69-73, 49.43): fc1 98304x64x8192 10.71 (HiFi4, 6.3 % FLOPs) - SiLU 7.08 - fc2 98304x8192x2048 26.27 -
  skip 98304x64x2048 2.67 - add 2.70. PAD on all; SiLU and the add are fusable (concat-K).
- **layer_time** (IDs 76-93, 170.60; x46): RMSNorm 1.93 - **QKV 98304x2048x6144 44.11** (18.4 % FLOPs, subblock 2x2) -
  create heads 5.75 (LAYOUT) - RoPE q 4.32 - RoPE k 4.33 - SDPA 4.34 - concat heads 1.96 (LAYOUT) - out_proj 7.22
  (37.5 %) - mul beta 2.23 - add 2.69 - RMSNorm 1.92 - **g 26.89 / u 26.59** (27 %, subblock 1x1) - SiLU 8.55 - mul
  g*silu(u) 7.20 - down 98304x5472x2048 15.65 (46.3 %) - mul beta 2.21 - add 2.70.
  Per layer: matmuls 120.46 ms (70.6 %), fusable eltwise (mul beta x2, add x2, SiLU, gate mul) 25.58 ms, attention core
  (create heads, RoPE x2, SDPA, concat) 20.71 ms, norms 3.85 ms. Every op runs on 98304 rows (PAD).
- **last_time** (IDs 96-117, 83.52): the time layer up to the residual add (IDs 96-105, 80.7), then the cut
  (untilize 1.89, slice 0.09, tilize 0.75 at 16x padding, reshape copy 0.46), then the MLP on 2048 rows (1.65).
  DEAD: the Q third of QKV, q heads, RoPE q, 94/96 of SDPA, concat and out_proj + residual (about 38 ms).
- **layer_variate** (IDs 120-132, 2.12): V = 1 shortcut v_proj 0.165 + out_proj 0.167 on 2048 rows, then the MLP.
- **head** (IDs 135-140, 1.25): fc1 0.58 - SiLU 0.16 - fc2 (fp32 out) 0.34 - skip 0.10 - add 0.02.

### Top 10 op roles (full model)

| # | role (IDs) | n | M x K x N | ms/call | full ms (%) | FLOPs % | notes |
|---|---|---|---|---|---|---|---|
| 1 | QKV + b (77, 97) | 47 | 98304x2048x6144 | 44.1 | 2073 (26.0) | 18.4 | auto 1D config; fp32 acc halves dest (subblock 2x2) |
| 2 | MLP g (87) | 46 | 98304x2048x5472 | 26.9 | 1237 (15.5) | 26.9 | subblock 1x1 |
| 3 | MLP u + SiLU requested (88) | 46 | same | 26.6 | 1223 (15.3) | 27.2 | SiLU runs as a separate unary (role 7) |
| 4 | down (91) | 46 | 98304x5472x2048 | 15.6 | 720 (9.0) | 46.3 | |
| 5 | RoPE q + k (79, 80, 99, 100) | 94 | (1024,32,96,64) | 4.3 | 407 (5.1) | - | HiFi4, 36 % DRAM |
| 6 | SiLU (89, 70, 113, 128, 137) | 50 | 98304x5472 | 8.55 | 401 (5.0) | - | |
| 7 | out_proj + b (83, 103) | 47 | 98304x2048x2048 | 7.2 | 339 (4.2) | 37.5 | |
| 8 | mul g*silu(u) (90, 114, 129) | 48 | 98304x5472 | 7.2 | 332 (4.2) | - | 87 % DRAM |
| 9 | create heads (78, 98) | 47 | (1024,1,96,6144) | 5.75 | 270 (3.4) | - | |
| 10 | residual add (85, 93, ...) | 96 | E = 98304x2048 | 2.7 | 251 (3.1) | - | 88 % DRAM |
| 11 | residual mul beta (84, 92, ...) | 96 | E | 2.2 | 209 (2.6) | - | removable exactly |

### Bottom line

- Device-bound (wall - device = 203 ms). The matmuls run at a third of their math floor on the auto 1D config
  (in0 re-read once per N block; analyst's fit: 1.27 us per in0 tile per core for every SLOW shape).
- 31.25 % of every per-token op is row padding; token 65 is dead too.
- 25.6 ms per layer of eltwise passes are removable by fusion; 2.2 ms per layer (the beta multiply) exactly.
- Savings overlap and are not additive. Gates: the PCC suite (72 + 1 skip, job 553 tests), every gate floor (`goldens.GATE_FLOORS`),
  target device >= 2 % better (about 160 ms), no guard regression > 3 %.

| # | Proposal | Finding | Est. full-model saving | Confidence | Risk / gate |
|---|---|---|---|---|---|
| R1-1 | Split context / decoder streams: context = tokens 0-62 + 1 pad slot (64, no row padding), decoder = the read-out tokens 63 and 64 packed 16 rows per tile with a block cross-mask; drop dead token 65 | F2 padding + dead token | -2.2 s today; about -0.9 s after R1-3..R1-7 | medium | high: micro-benchmark first, decoder attention <= 5 ms per layer and its error vs torch <= the current path's |
| R1-2 | Exact epsilon rescale: carry y = x / c (c = running product of the betas), RMSNorm epsilon eps / c^2, fold 1/c into out_proj / fc2: the beta multiplies disappear | F4 (fold) | -0.21 s (-2.6 %) | high | low (exact algebra; c = 0.686 over 96 updates) |
| R1-3 | Fused SwiGLU `minimal_matmul(fuse_swiglu=True)` (g, u, SiLU, mul in one op) and the SiLU fused into the embed / head matmuls | F3 | -0.70 s (-8.8 %) | high | low; sweep its blocks under fp32 acc (dest 4 tiles) |
| R1-4 | Residual add in the out_proj / down epilogue (`dit_minimal_matmul_addcmul_fused`), after R1-2 | F4 (epilogue) | -0.23 s (-2.9 %) | medium | needs weight dtype = activation dtype: the performance precision keeps the add |
| R1-5 | QKV split + head split + RoPE in two fused ops (`minimal_matmul_split`, Timestral's R3-1) | F5 | -0.34 s (-4.2 %) | medium | medium (off-label DiT op; head-major layout) |
| R1-6 | Matmul program configs per role (2D multicast / `minimal_matmul`), swept at the exact M / K / N under fp32 acc | F1 | -2.9 s (-36 %) | high on headroom, medium on the figure | low-medium (L1 limits; re-sweep after R1-1 changes M; guards need shape-derived configs) |
| R1-7 | Last-layer dead work without R1-1: cut before out_proj, cheaper slicing | F7 | -0.015 s | high | below the gate: bundle or drop if R1-1 lands |
| R1-8 | Host: per-patch sums instead of step-level cumsums in the scaler (prepare 85 ms) | F8 | wall only, about -0.1 s | medium | judge on wall in the same job (N2) |

Combined (analyst's composition, rescaled): R1-6 + R1-1 + R1-3 + R1-2/R1-4 together about -4.5 s (-57 %), not the
6.3 s naive sum.

**Round-1 order under the v1.7 loop (skill section 5.1 known winners first, screened on 313m with `BENCH_LAYERS=4`, kept ones confirmed on 2.5B-FT):** R1-3 fused SwiGLU (block sweep in one job), R1-6 matmul configs (one sweep job per role set), R1-5 QKV split + RoPE, R1-4 epilogue residual, R1-2; R1-1 after its micro-benchmark. Earlier recommendation, kept for the record: R1-1 (micro-benchmark first), then R1-2, R1-3, R1-4 and R1-6, in that order; R1-5
and R1-8 next round. Topology first (R1-1, R1-2), then fusion (R1-3, R1-4), then configs (R1-6) swept at the shapes
the topology leaves.

## R1-1: split context / decoder streams
- Hypothesis / mechanism: every per-token op runs on 96 slots per row for 66 tokens, and token 65 is never read
  (`tt/readout.py`: tokens P-1 .. P+h-2 are read; no read-out token attends to it under causal or group masks). Run
  the context tokens 0-62 plus one pad slot as rows of exactly 64 (2 tiles, causal SDPA on (R, H, 64, 64)) and the
  read-out tokens 63, 64 as a decoder stream of 2048 rows (16 rows per tile). Decoder queries attend over a block of
  16 rows' 64 context keys plus their 32 decoder keys (1056 keys) with a cached cross-mask (causal within the row,
  pad slot masked, group ids ANDed in with missing values). RoPE/xPos tables stay those of S = 66 (centre 33). The
  decoder stream is the head input, so the last layer's take_slots (IDs 106-109) and its dead work disappear, and its
  context stream computes only [Wk|Wv].
- Evidence: M = 98304 on IDs 69-105 (7981 of 7984 ms); 67584 real, 66560 live rows. Upper bound 0.3125 x 7984 =
  2495 ms plus 0.0104 x 7984 = 83 ms for token 65.
- Estimate (today's costs): per time layer 170.6 -> 2/3 x (170.6 - 4.34) + 2/3 x 4.34 (context SDPA on 64 keys)
  + 2.5 (decoder per-token ops at 2048 rows, from IDs 110-117) + 5.0 (decoder attention, Timestral's measured 0.62 +
  0.62 ms concat and 0.82-0.99 ms SDPA at 12 heads, x 32/12) = 121.2 ms: -49.4 x 46 = -2272 ms, last layer about -50,
  embed about -15: **about -2.3 s**. After R1-6 / R1-3 / R1-2 / R1-4 (layer about 75 ms): about -0.9 s.
- Rules: "Optimize the operation topology before fine-tuning individual op configs"; "keep padding, masking ... and
  output slicing inside the model path"; "Padding must be mathematically inert"; examples.md R4-1 ("look for padding
  and dead work in round 0").
- Plan: `tt/attention.py` (split_time with decoder attention), `tt/model.py` (two-stream inputs, cross-mask cache,
  split-aware embed / head), `tt/preprocessing.py` (per-stream tables). Copy Timestral's `SplitStreams` /
  `_cross_mask` / `_decoder_attention`, adapted to read-out tokens 63-64 and to separate q / k xPos tables.
- Experiment and gate: micro-benchmark the decoder attention alone at the paper shape (2048 decoder rows, 32 heads,
  1056 keys per block): <= 5 ms per layer and max error vs torch no worse than the current single-stream path's on the
  same inputs. Then path tests that force the split (paper-like rows of 64 + 2 tokens, with and without missing
  patches, V = 8 multivariate) and the full suite; long_context (288 = 9 tiles, no padding) keeps the single stream.
- Risks / fallback: decoder attention cost (+-1.5 ms per layer = +-70 ms); fallback (ii) packed 65-token rows with
  re-layout around attention, about -1.5 s.

## R1-2: exact epsilon rescale of the residual stream
- Mechanism: x_{k+1} = beta_k x_k + f_k. With y_k = x_k / c_k, c_k = prod_{j<k} beta_j: RMSNorm_eps(c y) =
  RMSNorm_{eps / c^2}(y) and y_{k+1} = y_k + f_k / c_{k+1}. Fold 1/c_{k+1} into W_o, b_o, W_fc2 (float64 in
  `tt/weights.py`); each norm gets its own epsilon (`ttnn.rms_norm` takes it per call). The residual becomes a plain add.
- Evidence: IDs 84, 92, 104, 116, 123, 131 (BinaryNg MUL beta): 2.23 + 2.21 ms per time layer; 209 ms per forward.
  betas (2.5B-FT): 0.9889-0.9947 attention, 0.9996-0.9998 MLP; c over all 96 updates 0.686, so y grows at most 1.46x
  and epsilon at most 5e-4 / 0.686^2 = 1.06e-3.
- Estimate: -209 ms (-2.6 %); exact in real arithmetic (bf16 rounding of y instead of x).
- Rules: "Avoid data movement before tuning math"; the topology audit's redundant-op removal.
- Plan: `tt/weights.py` (fold), `tt/common.py` (`scaled_residual` -> add), per-layer eps; the folded dataflow
  (`tests/golden/folded.py`) adopts it first and must stay within 1e-5 of the reference.
- Experiment and gate: dataflow test, the suite, bench >= 2 %; if below the gate, bundle with R1-4.
- Risks: program cache entries per distinct epsilon (cold start only).

## R1-3: fused SwiGLU and fused SiLU
- Mechanism: one `minimal_matmul(fuse_swiglu=True)` on a tile-interleaved [W_u | W_g] weight writes g * silu(u)
  (SiLU on the second half; Timestral's `upload_swiglu` with w_silu = W_u, w_mul = W_g) and drops the SiLU and MUL
  passes; zero d_ff padding (5464 -> 5472) stays inert. The embed / head SiLU goes into a 2D config's fused activation.
- Evidence: Unary SiLU IDs 70, 89, 113, 128, 137 = 400.7 ms (5.0 %); MUL g*silu(u) IDs 90, 114, 129 = 332 ms (4.2 %);
  `matmul.cpp` strips `activation=` without an explicit program config and runs it as a trailing unary.
- Estimate: -733 ms gross, less ~0.8 ms x 46 epilogue = **-0.70 s (-8.8 %)**; Timestral R2-1 measured -14 %.
- Rules: Matmul Choices "paired gate/up MLP projections ... one packed/fused projection"; OPT-010.
- Plan: `tt/mlp.py`, `tt/common.py` (`upload_swiglu`, `swiglu_config`); sweep M/K/N blocks with fp32 acc (dest limit
  4 tiles: subblock 2x2).
- Experiment and gate: block sweep at M = 98304 (and the M R1-1 leaves), then the suite, bench >= 2 %.
- Risks: low; check the performance precision (bfp8 weights) as Timestral did.

## R1-4: residual add in the matmul epilogue
- Mechanism: after R1-2 the residual update is y + f: `dit_minimal_matmul_addcmul_fused(h, W, 1.0, y, ones,
  bias_tensor=b)` on out_proj and down (Timestral's `linear_residual`, R2-3 -5.8 %).
- Evidence: residual ADD IDs 85, 93, 105, ... = 2.69 + 2.70 ms per layer, 251 ms per forward.
- Estimate: -251 + ~28 ms (residual read in the epilogue) = **-0.23 s (-2.9 %)**.
- Rules: fusion ("Prefer fused ... where possible"; topology audit).
- Plan: `tt/common.py` (`linear_residual`), `tt/attention.py`, `tt/mlp.py`.
- Experiment and gate: suite and bench; down must keep its matmul efficiency on `minimal_matmul`.
- Risks: requires weight dtype = activation dtype: performance keeps matmul + add (or bf16 wo / down, measured).

## R1-5: fused QKV split, head split and RoPE
- Mechanism: `minimal_matmul_split(chunks=3)` with bias, then `wan_fused_rmsnorm_post_allgather` (unit stats, eps 0)
  for the head split with RoPE on q and k (separate q / k xPos tables per token) and without it on v; SDPA on the
  (H, R, S_pad, Dh) view (Timestral R3-1).
- Evidence: create heads 5.75 + RoPE 4.32 + 4.33 = 14.4 ms per layer (IDs 78-80, 98-100), 677 ms per forward; RoPE
  runs at 36 % of DRAM peak.
- Estimate: Timestral's fused q/k/v at 957/954/801 us for 151 MB scaled x 2.67 -> 7.24 ms: -7.2 x 47 = **-0.34 s**.
- Rules: OPT-001 (no reshape / layout cluster to repair a split projection before rotary); Final Audit.
- Risks: medium (DiT op off-label; dim0 == 1 forces the head-major layout; per-token tables 12.6 MB each).

## R1-6: matmul program configs per role
- Mechanism: the auto 1D config (11x10 grid, mcast in1, per_core_M 28) re-reads in0 once per N block; the 2D
  multicast config gives each core its whole N slice (Timestral R1: -46 %). Port Timestral's `program_configs.py`
  (`matmul_2d_config`, `minimal_matmul_config`, shape-derived), sweep each role at the exact M / K / N with fp32 acc.
- Evidence: QKV ID 77 18.4 % FLOPs (44.1 ms), g / u IDs 87-88 27 % (26.9 / 26.6 ms, subblock 1x1), out_proj ID 83
  37.5 %, down ID 91 46.3 %, embed fc1 ID 69 6.3 %; `matmul_weighted_flops_percent` 31.29.
- Estimate at 55 % of the 110-core HiFi2 peak (167 TFLOPs): QKV 44.1 -> 14.8, g and u 26.9 / 26.6 -> 13.2 each, out_proj
  7.2 -> 4.9, down 15.6 -> 13.2: -61.0 ms per layer x 47 = **-2.9 s (-36 %)** (50 %: -2.7 s; 65 %: -3.2 s).
- Rules: "use 2D matmul program configs for large prefill matmuls"; "Explicitly configure ... program_config"; Matmul
  Choices (per-role search tables).
- Experiment and gate: `tests/perf/sweep_matmul.py` per role at the shapes in use (and at R1-1's M), with an auto
  control; the suite; bench >= 2 %; guards long_context (M = 9216) and paper_313m (d = 1024) must not regress > 3 %.
- Risks: L1 limits under fp32 acc; t0-gamma measured fp32 acc 9 % faster than bf16 acc once configured.

## R1-7: last-layer dead work (single stream)
- Cut before out_proj on the last time layer and slice in row-major: -16 ms + -1.2 ms; below the gate, part of R1-1.

## R1-8: host scaler
- prepare 85 ms per call on the host (float64 step-level cumsums over 1024 x 2112); per-patch sums (Timestral
  R3-5, 69 -> 25 ms) would cut wall by about 0.1 s. Wall-only: judge against a baseline re-run in the same job (N2).
