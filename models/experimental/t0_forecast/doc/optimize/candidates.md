# t0-gamma optimization candidates

Contract (`/home/andy/avan/fleet/t0-gamma/CONTRACT.md` v1, approved in D0):
- **Target:** `forward` (one `_single_pass`), `paper` shape: 1024 univariate series x context 2048 -> horizon 64
  (66 patches per row, 2 read out), `accuracy` precision (bf16 weights and activations, HiFi2 matmuls with fp32
  accumulation, norms at HiFi4 / fp32). Device time is primary (median of 5 `forward_device` calls, each after a
  0.3 s idle); steady wall time (median of 5 end-to-end calls) is secondary.
- **Guards** (none may regress by more than 3 %): `long_context-accuracy` (32 x 8192 -> 1024, 288 patches, 32 read
  out), `multivariate-accuracy` (128 series x 8 variates x 2048 -> 64, group attention over 8 rows),
  `paper-performance` (bfp8 weights, LoFi).
- **Floor:** the full PCC suite passes in one `pcc` job; target `pcc_normalized` >= 0.999 and `pcc_final` >= 0.99;
  `performance` guard >= 0.995 / 0.98; module tests >= 0.999; any candidate within 0.0005 of round 0's target
  `pcc_normalized` without an X gate.
- **Gate metrics (D1)**, on the read-out quantiles in asinh space, every bench record and suite case
  (`tests/helpers.quantile_metrics`): `pcc_normalized`, `pcc_spread` (PCC of `h - mean_q(h)`), `pcc_row_p01` and
  `pcc_row_min` (per-row PCC), plus `pcc_final`. Every metric scores the collapsed output (each quantile = its mean)
  below its floor on the realistic bench inputs. Proposed floors (`tests/helpers.FLOORS`, set by the orchestrator at
  G1): accuracy 0.999 / 0.998 / 0.995 / 0.98 / 0.99; performance 0.995 / 0.99 / 0.97 / 0.90 / 0.98; single
  0.999 / 0.999 / - / - / 0.99; single_long 0.9975 / 0.998 / - / - / 0.99; coverage cases (e2e, paths) 0.999 /
  0.998 / reported / reported / 0.99. Proposed drift vs round 0 on the target: normalized 0.0005, spread 0.002,
  row p01 0.002.
- **Precision exceptions (confirmed in D1):** fp32 dest accumulation on every matmul in both precisions (jobs 395,
  398); the patch encoder at HiFi4 (jobs 414, 417); the decoder output and monotone head in fp32 (job 383). Per-role
  sweep for the single-series cases: `iter_00/precision_sweep.json` (job 467), `precision_sweep_realistic.json` (job 475).
- **Gate:** accept at >= 2 % target device time (wall for host-only changes), no guard regressing by > 3 %, every
  floor holding: `compare_bench.py <cand> <best> --target paper:accuracy --pcc-key pcc_normalized --pcc-floor 0.995`,
  then the target floor by hand.
- **Exit:** two consecutive rounds each under 3 %, or matmuls > 65 % of FLOPs and non-matmul ops < 15 % of device
  time, or 6 rounds.
- **Commits:** one per accepted change, unsigned, one lowercase line, model paths only (PLAN.md section 7).

Headline: `forward`, `paper`, `accuracy`. Guards: `long_context-accuracy`, `multivariate-accuracy`, `paper-performance`.

| round | candidate | paper wall ms | paper device ms | vs best | guards (device ms) | PCC (norm / final) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline: auto 1D matmul configs, SwiGLU as gate + up + SiLU + MUL, residual adds separate, rows padded 66 -> 96 per row, read-out after layer 23; fp32 matmul accumulation, `embed` at HiFi4; realistic bench inputs (D1) | 1547.8 (host-loaded; prepare 363 ms) | 1056.8 | — | long 96.1, multi 2003.9, perf 1042.9 | norm 0.999969, spread 0.999955, row p01 0.999534, row min 0.997776, final 0.999986 (long 0.999899 / 0.999777 / 0.997717 / 0.997319; multi 0.999945 / 0.999954 / 0.999617 / 0.994025; perf 0.999328 / 0.999115 / 0.990240 / 0.946128) | baseline (replaces the job-420 row: smooth inputs, uncommitted tree) | iter_00/bench.json + summary.md (job 495 on 9c92695bdf6), PCC suite 47/47 (job 491) |
| 1 | R1-1: cut to the read-out patches after the last time attention (layer 22's MLP and layer 23 on R x n rows) | 1474.4 | 1009.2 (−4.5 %) | −4.5 % | long 91.1 (−5.2 %), multi 1840.5 (−8.2 %), perf 997.9 (−4.3 %) | metrics identical to round 0 (exact cut) | **accept** (bundle with R1-2); compare_bench ACCEPT, check_gates PASS | iter_01/candidates/r1_1_readout_cut.json (job 519), suite 50/50 (job 517), 553cd0d384b |
| 1 | R1-2: G = 1 group layers as one matmul, W = Wo Wv, b = Wo bv + bo | 1521.6 | 992.4 (−1.7 % vs R1-1) | −6.1 % vs round 0 (bundle) | long 91.2 (+0.1 %), multi 1840.4 (0.0 %), perf 980.7 (−1.7 %) | norm 0.999969, spread 0.999956, p01 0.999549, min 0.997749, final 0.999985 | **accept** as the D2 bundle (alone below the gate; no regression); check_gates PASS | iter_01/candidates/r1_2_fold_wvo.json (job 524), suite 50/50 (job 522), 9a7cec21ea4 |
| 1 | R1-3: context / decoder split streams (b3) — micro-benchmark gate only | – | – | – | – | decoder attention PCC vs torch 0.9999630 (current path 0.9999643) | **reject at the micro-benchmark gate** (D2: ≤ 2.0 ms per layer, PCC ≥ current): 2.27 ms per layer (2 concats of the context K/V ≈ 1.55 ms + SDPA over 1056 keys ≈ 0.72 ms); dropped, round 1 continues at the current M. At 2.27 ms the split would still save ≈ 250 ms: a concat-free layout goes to G2-r2 | iter_01/r1_3_micro.json (job 525), tests/perf/micro_decoder_attention.py (a8063130d1f) |
| 1 | R1-4: fused SwiGLU, one `minimal_matmul(fuse_swiglu=True)` (gate first, bias packed), blocks M 8 / K 16 / N 2, subblock 2 x 2 | 1098.0 | 796.1 (−19.8 %) | −24.7 % vs round 0 | long 76.1 (−16.5 %), multi 1646.2 (−10.6 %), perf 777.3 (−20.7 %) | norm 0.999969, spread 0.999955, p01 0.999484, min 0.997821, final 0.999987 | **accept**; compare_bench ACCEPT, check_gates PASS; per-call PCC vs unfused 0.999993 (accuracy and performance) | iter_01/candidates/r1_4_fused_swiglu.json (job 537), iter_01/swiglu_sweep*.json (jobs 528, 531), suite 50/50 (job 532), ace2296eff5 |
| 1 | R1-5: QKV on `minimal_matmul`, blocks M 16 / K 8 / N 4, subblock 4 x 1 (swept at M = 98304: 4.24 vs 12.67 ms per call) | 799.4 | 659.9 (−17.1 %) | −37.6 % vs round 0 | long 70.1 (−7.9 %), multi 1451.4 (−11.8 %), perf 625.9 (−19.5 %) | norm 0.999969, p01 0.999500, min 0.997821 (perf 0.999337 / 0.989821 / 0.952407) | **accept**; check_gates PASS | iter_01/candidates/r1_5_qkv_minimal.json (job 551), iter_01/matmul_sweep_qkv*.json, suite 50/50 (job 552) |
