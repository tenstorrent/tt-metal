# Round 1 proposals (from the round-0 analyst, mapped to the optimize skill)

Current best (round 0, `paper-accuracy`): **1201.0 ms** wall and **1056.4 ms** device, pcc 0.999998 / 0.999999
(`bench.json`, job 420). Reduced-profile estimate 1050.7 ms over 396 ops (**−0.5 %** vs measured; `summary.json`).
- **Where time goes** (full model, sections x counts): matmuls 609.9 ms (58.0 %), at 25.2 % weighted FLOPs on 110
  cores (304 TFLOPs HiFi2 basis). Every large matmul runs the auto 1D config (`per_core_M=28`, `in0_block_w=2`,
  subblock 2x2, `fused_activation=nullopt`). Non-matmul 440.8 ms move 162.7 GB, 72 % of their 317.8 ms DRAM floor,
  so they can only be removed or shrunk, not tuned.
- **Biggest structural costs:** rows pad 66 -> 96 tokens (31.25 % of every per-token op is padding, ≈ 328 ms row-scaled);
  the SwiGLU runs as 4 ops (gate, SiLU, up, MUL = 355 ms, 33.8 %); the QKV matmul alone is 207.9 ms (19.8 %) at 15.7 %
  of peak; layer 22's MLP and all of layer 23 (48.5 ms) run on 98304 rows of which 2048 are read.
- **Measurement caveats:** profiler signposts add ~3.7 ms of op-to-op gaps (excluded from device time; tracing is not
  worth anything here). The performance precision is only 1.1 % faster (1044.5 ms) because no matmul is math-bound yet.

`analysis.md` note: the analyst's full report could not be saved as `iter_00/analysis.md`, because the harness blocks
subagents (both the analyst and this worker) from writing analysis/report files. Its evidence is reproduced in the
sections below, with the profile row IDs (`#nn` = tt-perf-report `ID` in `perf_report.csv` / `sections/*.csv`).

Savings overlap and are not additive. Each estimate below is **after the items ranked above it** (the round-0-only
estimate is in brackets). Gates: the PCC suite (44 tests, job 457), target pcc_normalized ≥ 0.999 and within
0.0005 of round 0 (0.999998), target device time ≥ 2 % better, no guard regression > 3 %.

| # | Proposal | Finding | Est. savings (sequenced) [alone] | Confidence | Risk / complexity |
|---|---|---|---|---|---|
| R1-1 | Cut to the read-out patches right after layer 22's attention | E (c1) | 47.6 ms, 4.5 % [47.6] | high | low; exact by design |
| R1-2 | G = 1 group layers: one matmul with W = Wv·Wo, b = bv·Wo + bo | I | 16.9 ms, 1.6 % [19.3] | high | low; **below the gate alone: approve as a bundle with R1-1** |
| R1-3 | Context / decoder split streams (context 0..62 + 1 pad, decoder = read-out patches 63, 64; dead patch 65 dropped) | A (b3) | ≈ 266 ms, 27 % [286] | medium | high; **micro-benchmark first** (gate ≤ 2.0 ms per layer) |
| R1-4 | Fused SwiGLU: `minimal_matmul(fuse_swiglu=True)`, gate is the FIRST half | B | ≈ 140 ms [225] | high | medium (sweep blocks first) |
| R1-5 | QKV program config (2D multicast / `minimal_matmul` / `minimal_matmul_split(3)`) | C | ≈ 90 ms [136] | high | low (sweep first) |
| R1-6 | Residual add in the wo / down epilogue (`dit_minimal_matmul_addcmul_fused`) | H | ≈ 25 ms [37] | medium-high | low; accuracy precision only (weight dtype must match) |
| R1-7 | SDPA `output_concat_heads=True` in time attention | J | ≈ 9 ms [13] | high | low; **below the gate alone: bundle with R1-6** |
| R1-8 | Explicit configs for wo, down, embed, decoder (fused ReLU in the embed hidden) | F + G | ≈ 38 ms [57] | medium-high | low (sweep first) |
| R1-9 | Host `prepare`: vectorized scaler statistics, bit-identical | M | wall −60 to −80 ms, device 0 | medium | low; judged on wall |
| R1-10 | Multivariate group attention on whole 32-row tiles (4 series x 8 variates, block-diagonal mask) | §8 | guard only: multivariate ≈ −800 ms device; target 0 | medium | medium; **needs a decision: the target gate cannot pass** |

Projected sequence (not additive): 1056 -> ≈ 992 (R1-1, R1-2) -> ≈ 726 (R1-3) -> ≈ 586 (R1-4) -> ≈ 496 (R1-5)
-> ≈ 462 (R1-6, R1-7) -> ≈ 424 ms (R1-8), i.e. −60 %. Timestral went 1307 -> 354 ms over four rounds with the same
families of changes.

Not proposed this round: a fused per-head QK-RMSNorm + RoPE kernel (D: 112.3 ms today, ≈ 76 ms saving, but no
existing op normalizes per head; `wan_fused_rmsnorm_post_allgather` uses whole-row statistics; a model-local
`generic_op` kernel is a later-round candidate with its own micro-benchmark gate). RMSNorm tuning (L, ≤ 6.6 ms),
read-out layout moves (K, 1.6 ms, free with R1-3), tracing (J2, ≈ 0 device).

## R1-1: cut to the read-out patches after layer 22's attention

- **Hypothesis / mechanism.** Everything after layer 22's time attention is per token (MLP, G = 1 group layer) or per
  patch across a series' rows (G > 1 group layer), so it only needs the read-out patches 63 and 64 (2 of 96 slots).
  Move `readout` from after layer 23 to after layer 22's attention residual add; layer 22's MLP and layer 23 then run on
  M = 2048. The fp32 spec already does this (`tests/golden/dataflow.py`, `readout_cut=True`, checked at 1e-5 against the
  reference; `test_readout_cut_is_exact`).
- **Evidence.** `last` section #105–115 = 27.85 ms (layer 23 on 98304 rows); layer 22's MLP = time-layer #83–89
  = 20.68 ms. Read-out #116–119 (untilize, slice, tilize, reshape) = 1.57 ms stays, moved.
- **Estimate.** 20.68 + 27.85 = 48.54 ms removed, plus 18 ops back at M = 2048 at ~0.05 ms (head-section rows 22–68 µs)
  = **47.6 ms (4.5 %)**. Long context ≈ 4 ms (only 1024 of 9216 rows are read), multivariate exact (all G rows of
  patches 63, 64 kept; the group mask is sliced like the spec does).
- **Rules.** Core Optimization Rules: topology before configs, "count material matmuls ... in the measured path";
  examples.md dead work (Timestral R3-3).
- **Plan.** `tt/model.py`: `forward_device` / `run_layer` take the cut index (last time layer); readout of
  `TtAttentionInputs` for the cut stream (rows = R, padded_seq = n, group mask sliced). Profile sections become
  embed / layer_time / layer_group / cut (layer 22) / last (layer 23 on read-out rows) / head; `layer_counts.json`
  counts updated (layer_time 15).
- **Experiment and gate.** Full PCC suite (outputs should be identical up to bf16 order: per-token ops), bench target
  and guards. Accept at ≥ 2 % (alone or with R1-2).
- **Risks / fallback.** Group-mask slicing for G > 1 with masks (covered by `test_branch_paths`).

## R1-2: fold Wv·Wo in the G = 1 group layers

- **Hypothesis / mechanism.** With one row per series the group attention is `x + (xn Wv + bv) Wo + bo`; two
  back-to-back 98304 x 1024 x 1024 matmuls with nothing between them. Precompute W = Wv^T Wo^T and b = bv Wo^T + bo in
  fp32 at load: one matmul, one bf16 rounding instead of two. Multivariate (G > 1) keeps the full path.
- **Evidence.** #93 wv 2413.6 µs + #94 wo 2417.6 µs per group layer (x 8 in the full model).
- **Estimate.** 7 x 2.415 = **16.9 ms** after R1-1 (layer 23 is then on 2048 rows), 1.6 %: below the gate alone.
- **Rules.** Core Optimization Rules ("count material matmuls"); graph-fusing (merge adjacent ops).
- **Plan.** `tt/attention.py`: a `_wvo` / `_bvo` upload for GROUP layers, used when `group_size == 1`.
- **Experiment and gate.** PCC suite (group-layer module tests at G = 1 and the full model), bench. **Ask: approve
  R1-1 + R1-2 as a bundle** (≈ 64 ms, 6.1 %), each measured and recorded separately.

## R1-3: context / decoder split streams (variant b3)

- **Hypothesis / mechanism.** Each row has 66 patches padded to 96 slots; 31.25 % of every per-token op is padding.
  Patch 65's output is never read, and by causality nothing read depends on it, so it is dead. Split each row into a
  **context stream** (patches 0..62 plus one pad slot = 64 slots, two whole tiles) and a **decoder stream** = the
  read-out patches 63 and 64 (2 per row, 16 rows per 32-row tile). Per-token ops run on M = 65536 + 2048 = 67584
  rows (−31.25 %). Time attention: context queries attend causally within the context stream (the pad slot is last, so
  no real query reads it); decoder queries of a 16-row block attend over that block's concatenated context K/V
  (16 x 64) and decoder K/V (16 x 2) = 1056 keys through a block cross-mask (own row only, context pad slot masked,
  patch 64 sees 63). Group layers run on each stream separately (they mix rows within one patch slot). After layer
  22's attention the context stream is dropped (R1-1 becomes free) and layer 22's context needs only `[Wk|Wv]`
  (≈ 7.5 ms more, Timestral R4-3). The read-out *is* the decoder stream (no untilize / slice / tilize, K = 1.6 ms).
  The split applies only when it saves rows (Timestral's `split_rows_per_block` rule): long_context (288 = 9 tiles,
  no padding) stays single stream; multivariate splits too.
- **Evidence.** Every per-token row in all sections runs at 98304 rows (`98304 x ...`, `(1024,8,96,128)`); only
  1024 x 66 = 67584 are real. Row-scaled device time 1048.6 ms -> padding 0.3125 x 1048.6 = 327.7 ms.
  Variants rejected: b1 (drop patch 65 only: 65 still pads to 96, saves 0), b5 (flat 65-token packing: 32-row SDPA
  blocks over 2080 tiles, 14.7x the attention work).
- **Estimate.** After R1-1/R1-2 (≈ 992 ms base): 0.3125 x ≈ 985 = 308 ms gross, minus decoder attention ≈ 16 x 1.5 =
  24 ms, minus ≈ 19 ms of duplicated small decoder-stream ops (16 x 18 + 8 x 11 ops at ~0.05 ms), plus 1.6 ms of
  read-out moves = **≈ 266 ms (27 %)**. Timestral R4-1 measured −21.7 % at the same 64 + 2 geometry.
- **Rules.** Core Optimization Rules: "Optimize the operation topology before fine-tuning individual op configs",
  "Padding ... is a common source of bugs or wasted work"; OPT-005 (keep the logical batch separate from tile
  padding); examples.md ("look for padding and dead work in round 0").
- **Plan.** `tt/model.py` (split upload: context tokens, per-stream RoPE tables and masks, the constant cross mask),
  `tt/attention.py` (`split_time`: both streams' QKV, decoder attention over concatenated keys; group layers per
  stream). Copy the pattern from Timestral `tt/model.py` / `tt/attention.py` (`SplitStreams`, `_cross_mask`,
  `_decoder_attention`), adapting to t0's per-head QK-norm + xPos tables (q and k tables per stream).
- **Experiment and gate.** **Micro-benchmark first**: the decoder-stream attention of one layer at the paper shape
  (concat of the context K/V, SDPA over 64 blocks of 32 queries x 1056 keys, cross mask) must take **≤ 2.0 ms per
  layer**, with PCC vs torch ≥ the current path's on the same layer inputs. Then PCC tests whose shapes force the split
  (with an assertion that the split path was taken), including the masked inputs (`e2e`, `paths`) and multivariate on
  both streams, the full suite, and the bench.
- **Risks / fallback.** Mask correctness with PAD patches (non-clean inputs need a per-row cross mask, as Timestral's
  `_cross_mask`); G > 1 group attention on both streams; the context pad slot must stay finite. Fallback: the single
  stream for shapes that do not split.

## R1-4: fused SwiGLU

- **Hypothesis / mechanism.** One K = 1024, N = 4096 `minimal_matmul(..., fuse_swiglu=True)` emits `silu(gate) * up`
  (width 2048) and replaces gate + SiLU + up + MUL. Pack w1 and b1 with
  `models.tt_dit.utils.tensor.prepare_for_fused_swiglu(w, ndev=1, gate_is_first=True)`: **t0's gate is the first half**
  (`gate, x = chunk(2)`; Toto and Timestral gate the second).
- **Evidence.** Per time layer #84 gate 4836.4 + #85 SiLU 2332.4 + #86 up 4832.7 + #87 MUL 2697.6 = 14.70 ms; full
  model 116.17 + 57.76 + 116.07 + 64.99 = **355.0 ms (33.8 %)**. The SiLU requested through `ttnn.linear(activation=)`
  is not fused under the auto config (gcc 80896 `fused_activation=nullopt`). The four ops move 16 E (3.2 GB) per layer.
- **Estimate.** 2 x 98304 x 1024 x 4096 = 824.6 GFLOP per layer; at 50 % of 304 TFLOPs, 5.43 ms (Timestral reached
  54 %). Round-0 saving 355.0 − 24 x 5.43 = 224.7 ms; after R1-1..R1-3 (22 layers at M = 67584, 2 at 2048):
  22 x (14.70 − 5.43) x 0.6875 ≈ **140 ms**. The config-only alternative (2D gate/up with fused SiLU, MUL kept) saves
  ≈ 148 ms at round 0, so the fused op wins by ≈ 77 ms; the packed-vs-split comparison is recorded (OPT-010).
- **Rules.** Matmul Choices ("paired gate/up MLP projections ... packed/fused projection"; "For 3-matmul MLP
  gate/up/down groups, explicitly compare ..."); OPT-010; graph-fusing.
- **Plan.** `tt/mlp.py` (one packed weight / bias, `minimal_matmul(fuse_swiglu=True)` with a `MinimalMatmulConfig`);
  a block sweep `tests/perf/sweep_swiglu.py` at the exact M (Timestral's pattern).
- **Experiment and gate.** Sweep M / K / N blocks and subblocks at the post-R1-3 M (fp32 dest acc limits the DST to 4
  tiles); per-call PCC vs the unfused path ≥ 0.9999; then the suite and the bench. Check the performance precision
  (bfp8 w1) separately.
- **Risks / fallback.** A wrong `gate_is_first` passes shapes and fails PCC (the MLP module test catches it); L1 limits
  on large blocks.

## R1-5: QKV program config

- **Hypothesis / mechanism.** The auto 1D config split the 96-tile-wide output into 8 N-blocks (`per_core_N=96`,
  `out_block_w=12`, `out_block_h=14`), so every core re-reads its in0 rows 8 times. A 2D multicast config over the grid
  (or `minimal_matmul` / `minimal_matmul_split(chunks=3)`) reads in0 once per grid row.
- **Evidence.** #73 98304 x 1024 x 3072 = 12993.9 µs at **15.65 % FLOPs / 12.2 % DRAM** (gcc 69632), 25.5 % of a time
  layer, 207.9 ms in the full model; ff1 #84 at the same K reaches 28.0 %. Timestral's sweep at the same M: auto 5.99
  -> 2D 2.73 ms (2.2x); its split reached 43.7 %.
- **Estimate.** At 45 % of peak, 618.5 GFLOP -> 4.52 ms per call: round-0 saving 16 x 8.47 = 135.6 ms (range 91.6 at
  28 % to 142.8 at 50 %); after R1-3 (M = 67584): ≈ **90 ms**.
- **Rules.** Core Optimization Rules ("use 2D matmul program configs for large prefill matmuls", "Explicitly configure
  ... program_config"); Matmul Choices ("local program-config search is required ... Tune QKV ... separately",
  "in0_block_w>=2 is only a floor", "Leave a table for every dominant matmul search").
- **Plan.** `tt/program_configs.py` (new) + `tests/perf/sweep_matmul.py`; candidates: 2D mcast on 11x10 and 8x10
  (N = 96 tiles divides), `minimal_matmul`, `minimal_matmul_split(3)` (which also feeds the head split).
- **Experiment and gate.** Sweep at the exact post-R1-3 M with a search table (config, ms, PCC vs the auto config);
  then the suite and the bench.

## R1-6: residual add in the wo / down epilogue

- **Hypothesis / mechanism.** `dit_minimal_matmul_addcmul_fused(ctx, Wo, 1.0, x, ones, bias_tensor=bo)` computes
  `x + ctx Wo + bo` in the matmul epilogue (reads the residual once instead of a separate 3 E ADD); same for w2.
  Biases are already fused (no bias ADD rows).
- **Evidence.** 48 residual ADDs at 1.336–1.356 ms (#82, #89, #95, #102, #108, #115) = 64.7 ms, 88 % of their 3 E floor;
  embed ADD #69 1.35 ms (out + skip could be one K-concatenated matmul, Timestral R4-5).
- **Estimate.** Timestral netted 56 % of the ADD time: 0.56 x 64.7 x 0.69 (after R1-3) ≈ **25 ms**.
- **Rules.** graph-fusing (fold adjacent ops into an existing op); Matmul Choices on dtype constraints for fused paths.
- **Risks.** The fused op needs weight dtype == residual dtype, so the performance precision keeps matmul + ADD unless
  wo / w2 stay bf16 there (Timestral R3-4: a separate decision).

## R1-7: SDPA `output_concat_heads=True` (time attention)

- **Evidence.** #80 `nlp_concat_heads` 920.2 µs after #79 SDPA (`output_concat_heads=false`); 16 x 0.92 = 14.7 ms. The
  group path already uses it.
- **Estimate.** Timestral R2-4 kept 90 %: 0.9 x 14.7 x 0.69 ≈ **9 ms**: below the gate alone; **bundle with R1-6**.
- **Rules.** Core Optimization Rules (SDPA contracts); graph-fusing.

## R1-8: explicit configs for wo, down, embed and decoder

- **Evidence.** wo / wv 98304 x 1024 x 1024 at 2.416 ms, 28.1 % (32 calls); down 98304 x 2048 x 1024 at 3.618 ms, 37.5 %
  (24 calls); embed hidden / skip K = 96 at `in0_block_w=1` (#65 1373.7, #68 1327.5 µs, 9.3–9.6 %), embed RELU #66
  1048.2 µs unfused; all `in0_block_w=2` with legal 4..32.
- **Estimate.** wo 32 x (2.416 − 1.40) = 32.5 ms, down 24 x (3.618 − 2.71) = 21.7 ms, embed 1.9 ms, ReLU 1.0 ms at round
  0; ≈ 38 ms after R1-2/R1-3 (wv gone, M scaled).
- **Rules.** Matmul Choices (per-role search; "in0_block_w>=2 is only a floor"; subblock call-out; search tables);
  Compute Kernel Configs (fp32 dest acc limits the DST: block shapes are the lever, not the subblock size).
- **Advice policy.** tt-perf-report's "HiFi2 may also work" on the HiFi4 embed rows is rejected with evidence (job 417:
  long_context pcc 0.99870 at HiFi2). "Place input 0 in L1" is infeasible (201 MB activations).

## R1-9: host prepare (wall only)

- **Evidence.** Wall − device = 144.6 ms: prepare 105.7, upload 14.4, postprocess 12.9 ms (bench.json medians).
- **Plan.** Vectorize the scaler / patching / mask prep (float64 per-patch sums bit-identical to the per-step Welford
  at fp32, as Timestral R3-5: 69 -> 25 ms); keep `test_scaler_and_buffer` at rtol 1e-5.
- **Gate.** Judged on wall and the host split (device unchanged), ≥ 2 % wall.

## R1-10: multivariate group attention on whole tiles (guard only)

- **Hypothesis.** Multivariate is +947 ms over paper (2003.7 vs 1056.4 ms), all in the 8 G = 8 group-attention
  sublayers (≈ 118 ms each). Reshaping to (S·B, G = 8, 3d) pads every 8 rows to a 32-row tile (4x the bytes) before
  `fill_implicit_tile_padding`, the head split, the QK norms and SDPA; plus two tiled permutes. Running SDPA on
  (S·R/32, H, 32, hd) tiles (4 series x 8 variates each) with a block-diagonal mask removes the 4x padding.
- **Evidence.** Reasoned from `attention.py:_group` (byte floors ≈ 48–53 ms per layer vs ≈ 118 ms measured); the
  multivariate shape was not profiled (next round: `T0_PROFILE_SHAPE=multivariate`).
- **Estimate.** ≈ 100 ms per group layer, ≈ −800 ms (−40 %) on multivariate; **0 on the target**.
- **Ask.** The target gate cannot pass for a guard-only change: approve it explicitly (with "accept on multivariate
  device ≥ 10 %, target unchanged within noise"), or defer.
