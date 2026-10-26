# Round 1 proposals (from the round-0 profile `iter_00/`, mapped to the optimize skill)

This file is the round's analysis record (PLAN section 9): the preamble carries the analyst's evidence, checked
against `summary.json` and `bench.json`; the read-only analyst subagent's report is not stored separately.

## Round 0 evidence

**G1 re-raise (D1): the accuracy policy now uses fp32 dest accumulation** on every matmul and both SDPAs (sweep
variant A1, `accumulation_sweep.md`), and the performance policy is bf8 except the FFN (bf16) + LoFi + fp32
accumulation (P2a). Round 0 was re-measured on HEAD 81ad9734042 (job 536); the first round 0 (HiFi2 without fp32
accumulation, job 497) is kept below for reference. The op-level evidence and the proposals are unchanged except
where noted (the matmul rows moved: QKV 16.41 -> 22.11 ms, the 1280^2 matmuls 4.4-4.6 -> 3.07-3.09 ms,
W_o 5.02 -> 3.41 ms; the layer total is the same, 67.85 ms).

**Headline** (target `forward`, `paper` = 1024 x 2048 -> 64, `accuracy`; realistic bench goldens, chip
61C0BD39B27E9772, job 536): device **1372.0 ms**, steady wall **1971.5 ms** (prepare 348.1, upload 201.6,
postprocess 21.5 ms; the host split is contention-sensitive across jobs). Reduced-layer estimate 1368.4 ms
(embed 9.37 + 20 x 67.85 + head 2.15), **-0.3 %** against measured. Every gate holds on every case
(`bench.json` `floor_ok`).

| case | device ms | wall ms | pcc_normalized / spread | row p01 / min | active-row p01 / min | spread ratio p01 / p99 (min / max) | pcc_final / final_spread |
|---|---|---|---|---|---|---|---|
| paper / accuracy (target) | 1372.0 | 1971.5 | 0.99993 / 0.99988 | 0.830 / 0.688 | 0.9985 / 0.9926 | 0.930 / 1.033 (0.689 / 1.115) | 0.999997 / 0.99993 |
| paper / performance | 1114.7 | 1785.9 | 0.99971 / 0.99916 | 0.206 / 0.139 | 0.977 / 0.823 | 0.890 / 1.065 (0.297 / 1.115) | 0.99999 / 0.99922 |
| long_context / accuracy | 232.0 | 367.2 | 0.99996 / 0.99992 | 0.864 / 0.856 | 0.9995 / 0.9995 | 0.990 / 1.016 (0.990 / 1.017) | 0.999996 / 0.99995 |
| multivariate / accuracy | 5110.0 | 5880.6 | 0.99994 / 0.99992 | 0.999 / 0.995 | 0.9992 / 0.9971 | 0.978 / 1.030 (0.840 / 1.051) | 1.000000 / 0.99992 |

First round 0 (superseded; HiFi2 without fp32 accumulation, bf8 + LoFi performance; job 497): paper 1370.5 ms,
pcc_normalized 0.99987, active-row min 0.949, spread ratio min 0.397 (rows 458 / 483 at 0.40 / 0.48);
performance 1369.7 ms with negative per-row PCC; long_context 180.8 ms; multivariate 4998.1 ms. The policy change
costs long_context +28 % and multivariate +2.2 % under the default 1D configs (fp32 accumulation halves DST);
R1-4 re-tunes blocks and subblocks for it. Both guards are re-baselined.

**Category split** (first round 0, scaled by the layer counts; the re-measured profile has 23.4 % weighted matmul
FLOPs, `summary.json`): matmuls 707.0 ms (51.6 %) at a weighted 22.4 % FLOPs; non-matmul 663.2 ms (48.4 %),
which already runs at ~70 % of its DRAM floor (465.8 ms) except RoPE (149.9 ms against a 47.2 ms floor).
Exit criteria (matmul > 65 % FLOPs and non-matmul < 15 %): not met.

**Op-to-op map, one layer (first round 0: 67.86 ms; IDs from `perf_report.txt`)**, all ops on 110
cores, DRAM interleaved, bf16. 32 of the 96 slots are dead (30 tile padding + the 2 never-read horizon tokens),
and 16 of 96 head lanes are zero:

| ID | op | ms | note |
|---|---|---|---|
| 43 | pre-norm (seq) | 1.21 | |
| 44 | QKV 98304 x 1280 x 4608 | 16.41 | 1D default config, `in0_block_w` 2, subblock 7x1, activations re-read 8x; 23.2 % FLOPs |
| 45 | create heads | 4.79 | layout move |
| 46, 47 | RoPE q, k (rotary_embedding_llama) | 3.75 + 3.75 | "reload" path: re-reads cos/sin every 3 tiles |
| 48, 49 | QK RMSNorm q, k | 1.43 + 1.42 | |
| 50 | SDPA causal | 2.66 | |
| 51 | concat heads | 1.41 | removable: SDPA `output_concat_heads` |
| 52 | W_o 98304 x 1536 x 1280 | 5.02 | 25.3 % FLOPs |
| 53, 54 | post-norm, residual add | 1.27 + 1.67 | |
| 55 | pre-norm (variate) | 1.21 | redundant (scale invariance, R1-2) |
| 56 | W_ov 98304 x 1280 x 1280 (V = 1) | 4.38 | |
| 57, 58 | post-norm, add | 1.27 + 1.65 | |
| 59 | pre-norm (FFN) | 1.21 | redundant (R1-2) |
| 60, 61 | ff0 + standalone ReLU | 4.56 + 1.31 | ReLU not fused (no program config) |
| 62 | ff1 | 4.57 | |
| 63, 64 | post-norm, add | 1.26 + 1.67 | |

Embed (11.08 ms): two K = 192 matmuls at 8.5 % FLOPs (1.88 ms each), a standalone ReLU, the 1280^2 matmul and an
add. Head (2.15 ms): an untilize / slice / tilize / reshape chain around a 45 µs matmul.

**Top-10 op roles** (full model): QKV 328.1 ms; RoPE 149.9; residual add 101.3; W_o 100.5; create heads 95.8;
ff1 91.4; ff0 91.1; W_ov 87.5; post-norms 75.9; pre-norms 72.5 (87 % of the device time together). Every matmul
row is SLOW at 23-25 % FLOPs and 14-23 % DRAM, and the `performance` policy (LoFi, bfloat8_b) gives the same
device time (1369.7 vs 1370.5 ms): the matmuls are limited by the default 1D data path, not by math or weights.

**Measured before this round** (exploratory, same code, job ids in STATUS):
- dropping the two trailing horizon tokens: paper 1371 -> 1147 ms (-16.3 %), multivariate 4998 -> 3647 ms
  (-27 %), head output bit-identical (PCC 1.0); exact in fp64 (`dataflow_report.json`);
- an fp32 residual stream: +17.6 % device for +0.0003 stack-output PCC (not adopted);
- patch encoder at HiFi4 or HiFi4 + fp32 accumulation (PLAN 9): no measurable change on long_context or paper
  (pcc_normalized 0.99991 / 0.99991 / 0.99991, embed 2.71 / 3.06 / 2.77 ms; job 483); kept at HiFi2;
- the `performance` policy's per-row failure comes from bfloat8_b FFN weights: keeping ff0/ff1 in bf16 lifts
  pcc_spread 0.9933 -> 0.9989 and row p01 -0.54 -> +0.24 on paper[:256]; bf16 tokenizer, QKV, W_o or W_ov
  alone change nothing, and bf8 + HiFi2 is as bad as bf8 + LoFi (bf16 + LoFi: spread 0.9992).

Savings overlap and are not additive. Gates: the PCC suite (57 tests), every gate of `golden_io.GATES` and the
drift band (`tests/perf/check_gates.py`), target device improvement >= 2 %, no guard regression > 3 % on device
time. **Order approved in D2** (effective when G1 passes): R1-1, R1-2 + R1-3, R1-4, R1-7, R1-5, R1-6; R1-8 deferred
until a multivariate profile gives op IDs.

| # | Proposal | Finding | Est. savings (target, device) | Confidence | Risk / complexity |
|---|---|---|---|---|---|
| R1-1 | drop the 2 trailing horizon tokens (66 -> 64 tokens, S 96 -> 64) | dead work + tile padding | 224 ms measured (-16 %) | high | low |
| R1-2 | drop the redundant gain-free pre-norms before W_ov (V = 1) and ff0 | redundant ops | ~32 ms (2.4 %) | medium-high | low (fp64 proof first) |
| R1-3 | SDPA `output_concat_heads=True`, drop the concat-heads op | layout move | 9-19 ms (0.8-1.6 %) | high | low; below the gate alone |
| R1-4 | matmul program configs per role (2D multicast / minimal_matmul), ReLU fused | 1D default configs at 23 % FLOPs | 240-520 ms after R1-1 | medium-high | medium (sweep first) |
| R1-5 | last-layer cut: layer 20 runs only the read-out queries after K/V | dead work | 20-38 ms | medium-high | medium |
| R1-6 | RoPE on a leading-dim view that avoids the reload path | 3x cos/sin re-reads | 25-37 ms | low-medium | low (micro-benchmark first) |
| R1-7 | performance policy: the cheapest variant passing the performance gates, after R1-4 (interim: P2a) | per-row metrics (bf8 FFN) | guard accuracy; device cost after R1-4 | high (accuracy) | low |
| R1-8 | multivariate variate attention without the 4x tile inflation (guard) | V = 8 padded to 32 rows, 2 permutes | >= 40 ms / layer on the guard | medium | medium (micro-benchmark first) |

## R1-1: drop the 2 trailing horizon tokens
- Hypothesis / mechanism: for every horizon the last `ROLLS = 2` horizon tokens come after the last read-out token;
  sequence attention is causal and the variate sublayer and FFN are per patch, so no read-out depends on them,
  and their positions (absolute RoPE) and the leading-pad masks are unchanged by dropping them. At H = 64,
  66 -> 64 tokens removes a whole tile row (S 96 -> 64): one third of every per-token op.
- Evidence: the head slice (ID 68) reads token 63 only; IDs 43-64 all carry 96 slots for 64 useful tokens;
  fp64 dataflow check "drop the 2 tokens after the last read-out": rel 0.0 (16 x 2048 -> 64), 5.9e-15 (-> 128),
  0.0 (4 x 8 variates); device run: 1371 -> 1147 ms, head PCC 1.0.
- Arithmetic: measured -224 ms (-16.3 %), less than the -33 % of rows, so the 1D matmul configs probably degrade
  at M = 65536 (`per_core_M` 19, prime): re-profile at S = 64 before R1-4's sweep. long_context: 512 -> 510
  tokens, S stays 512 (no change, no harm). multivariate: -27 %.
- Optimize-skill rules: operation-topology audit; topology before knobs; padding is wasted work.
- Plan: `upload_inputs` uploads tokens 0 .. readout_start + readout_count - 1 only (RoPE tables, masks and the
  read-out slot follow the kept count); add a suite test asserting the dropped count, on shapes that take the
  path (H = 64 and H = 128).
- Experiment and gate: full suite, then the benchmark; accept on >= 2 % (expected -16 %).
- Risks / fallback: none known; the change is exact.

## R1-2: drop the redundant pre-norms before W_ov (V = 1) and ff0
- Hypothesis / mechanism: with s = 1/rms(h) per token, post_norm(W_ov (s h)) = post_norm(W_ov h) and
  post_norm(ff1 relu(ff0 (s h))) = post_norm(ff1 relu(ff0 h)): RMSNorm is scale-invariant per row, ReLU is
  positively homogeneous, and there are no biases (gains are folded into the weights). Only the eps terms
  (1.19e-7) differ. Not valid for the sequence sublayer (V carries the per-token scale) or the V > 1 variate
  path (its values), so the multivariate path keeps its variate pre-norm and drops only the FFN one.
- Evidence: IDs 55 and 59, 1.206 + 1.208 ms per layer; their consumers (56 -> 57, 60 -> 61 -> 62 -> 63) end in a
  post-RMSNorm.
- Arithmetic: 2.414 ms x 20 = 48.3 ms at S = 96, x 2/3 = ~32 ms at S = 64 (2.4 % of 1147 ms).
- Rules: operation-topology audit (redundant ops).
- Plan: an fp64 dataflow check of the rewrite first (`check_dataflow.py`), then the device change.
- Experiment and gate: >= 2 % on its own expected; if R1-3 is also below the gate, measure R1-2 + R1-3 as one
  bundle (two stock-op removals) and raise it as a bundle if needed.
- Risks: bf16 matmul inputs keep the unnormalized residual's outlier channels (|x| up to ~1000); the relative
  precision is the same. Check every gate metric and the -0.0005 drift band.

## R1-3: SDPA writes the heads concatenated
- Hypothesis: `output_concat_heads=True` makes SDPA write [b, 1, s, 16 x 96] itself, so concat heads (ID 51)
  disappears.
- Evidence: ID 51 1.406 ms per layer (a pure 2Hd layout move); the SDPA row shows `output_concat_heads=false`.
- Arithmetic: 28.1 ms at S = 96, ~19 ms at S = 64, minus any slower SDPA write: 9-19 ms (0.8-1.6 %).
- Rules: Core Optimization Rules (use the composite SDPA op); Final Audit (no unnecessary layout ops).
- Experiment and gate: below the 2 % gate alone; measured and decided together with R1-2.
- Risks: check the variate path's masked SDPA accepts it too.

## R1-4: matmul program configs per role, ReLU fused
- Hypothesis: the default 1D config (`in0_block_w` 2, per-core full-N blocks, a 7x1 QKV subblock, weights
  multicast from one row) leaves the FPU ~23 % busy; a 2D multicast config over the 11 x 10 grid or
  `minimal_matmul` reads activations once with larger K blocks. An explicit config also fuses the ReLU into
  ff0 and the tokenizer's hidden matmul (packer ReLU), removing IDs 61 and 37.
- Evidence: IDs 36-62 matmul rows all SLOW at 8.5-25.3 % FLOPs; the performance policy (2x math peak, half the
  weight bytes) gives the same device time; Timestral's identical change on the same chip gave 2.3-4.7x per
  role (`models/experimental/timestral/doc/optimize/iter_01/`).
- Arithmetic: 2.51 TFLOP per layer at S = 96 (1.675 at S = 64); at 40-55 % of 303.8 TFLOPs that is 10.0-13.8 ms
  per layer at S = 64 against ~23-35 ms now: 190-500 ms, plus the fused ReLU (~18 ms) and the embed (~3 ms):
  **240-520 ms** after R1-1.
- Rules: Matmul Choices (2D multicast for large-M prefill; `in0_block_w >= 2` is only a floor; subblocks >= 2x1;
  a search table per dominant role); Prefill Latency (tuned 2D matmul, `minimal_matmul`).
- Plan: `tests/perf/sweep_matmul.py` at the exact post-R1-1 shapes (M = 65536 paper / multivariate, 16384
  long_context; roles QKV, W_o, W_ov, ff0 + ReLU, ff1, tokenizer) and both precisions, timing and PCC against
  torch per config; then a per-(M, K, N) config helper; LoFi vs HiFi2 re-checked after the configs change.
- Experiment and gate: sweep table first; then the suite and the benchmark; >= 2 % (expected -20 to -45 %), and
  long_context must not regress > 3 % (its M is 4x smaller).
- Risks: L1 circular-buffer limits with bf16 weights; longer cold compile.

## R1-5: last-layer cut
- Hypothesis: layer 20 needs keys and values for every token but queries, W_o, both post-norms, the variate
  sublayer and the FFN only at the read-out tokens (1 per row at H = 64, 31 at H = 1024).
- Evidence: IDs 51-64 of layer 20 (32.4 ms at S = 96) feed only the head's 1-of-96 slot; fp64 dataflow check
  "last layer cut to the read-out queries": rel <= 2.7e-15.
- Arithmetic: ~45 ms at S = 96; 25-38 ms after R1-1; ~20-25 ms after R1-4 (K|V projection cheaper).
  long_context ~4-5 ms; multivariate ~110-140 ms (its V = 8 path at 1 of 64 slots).
- Rules: operation-topology audit (dead work).
- Plan: a last-layer path: K|V for all tokens, Q for the read-out slots, a query-sliced SDPA (the leading-pad
  mask rows sliced too), the rest on M = rows x F.
- Experiment and gate: suite (with a test that takes the path for F = 1 and F > 1), benchmark, >= 2 %
  (expected 1.7-3 %; may need bundling).
- Risks: code complexity; the multi-token causal block at long_context.

## R1-6: RoPE without the reload path
- Hypothesis: `rotary_embedding_llama` picks its reload implementation when a core holds more than 8 rows
  (3 seq tiles x 16 heads = 48), re-reading 3 cos and 3 sin tiles every step; on a free leading-dim view
  (1024, 16, 64, 96) -> (4096, 4, 64, 96) each core holds 8 rows and reads cos/sin once per batch.
- Evidence: IDs 46/47, 3.75 ms each, moving 1208 MB per call (2.36 ms floor) for 604 MB of logical traffic.
- Arithmetic: ~0.9 ms per call at the same efficiency: 25-37 ms per forward.
- Rules: the skill's "read the op's code when a ttnn limitation appears"; also a tt-metal improvement candidate
  (the hard-coded threshold).
- Experiment and gate: micro-benchmark first at the post-R1-1 shape: <= 2.0 ms per call, output identical to the
  current op; then the suite and the benchmark.
- Risks: the per-step overhead, not bandwidth, may dominate; then reject.

## R1-7: performance policy: bf16 FFN weights
- Hypothesis: the bfloat8_b block exponent on ff0/ff1 adds a systematic error to every row; it is invisible in
  pcc_normalized (0.9979) but drives per-row PCC negative on flat rows and pcc_spread to 0.9938.
- Evidence: the per-role experiment above (paper[:256]): only bf16 FFN weights fix it (spread 0.9989, row p01
  +0.24 vs -0.54); bf8 + HiFi2 is as bad as bf8 + LoFi.
- Arithmetic: device cost unknown until R1-4 (bf8 currently saves nothing); expected small (FFN weights are
  3.3 MB per matmul against ~250 MB of activations).
- Rules: Precision And Fidelity (tune one tensor group at a time, with real weights and recorded inputs).
- Experiment and gate: after R1-4, measure the performance guard with ff0 / ff1 / both in bf16; keep the
  cheapest variant whose row metrics pass the floors set at G1. Contract: "documented per-role exceptions
  allowed".
- Risks: none for accuracy; may cost a little performance-guard time.

## R1-8: multivariate variate attention without tile inflation (guard only)
- Hypothesis: the V = 8 path reshapes to (S x B, 8, 4608), which pads every 8-row block to a 32-row tile (4x the
  bytes) before head split, QK-norm and SDPA, plus two full permutes. Packing 4 items x 8 variates of one slot
  into each 32-row tile with a block-diagonal 32 x 32 additive mask removes the inflation and the fill.
- Evidence: multivariate 4998 ms vs paper 1370 ms on the same 1024 rows x 96 slots: ~181 ms per layer for the
  general variate path; not profiled this round (profile the guard first).
- Arithmetic: >= 40 ms per layer, >= 800 ms on the guard. Target unaffected.
- Rules: Core Optimization Rules (padding is wasted work); operation-topology audit.
- Experiment and gate: micro-benchmark first: one layer's variate sublayer <= 40 % of the current one,
  PCC >= the current path's against torch; the masked SDPA must cover the pad keys explicitly (see the
  SDPA finding in GATE.md).
- Risks: SDPA's handling of a user mask with padding; per-variate padding (only in module tests).

## Not proposed this round
- A fused head-split + RoPE + QK-norm kernel (`generic_op`, model-local): ~146 ms after R1-1, large effort;
  after R1-4 shows what remains.
- A fused post-norm + residual add kernel: ~49 ms after R1-1, no stock op.
- Tokenizer packing (fold the 128 constant token features into a bias, K-concat): 2.5-4 ms.
- A tile-aligned read-out gather: ~1 ms (folds into R1-5).
- Host: the token tensor (37.7 MB at S = 96) shrinks 4.5x with R1-1 plus the constant-feature fold; the host
  split is contention-sensitive (prepare 162 -> 369 ms between two runs); a later round, judged on wall.
