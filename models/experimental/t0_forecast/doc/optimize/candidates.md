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
- **Gate:** accept at >= 2 % target device time (wall for host-only changes), no guard regressing by > 3 %, every
  floor holding: `compare_bench.py <cand> <best> --target paper:accuracy --pcc-key pcc_normalized --pcc-floor 0.995`,
  then the target floor by hand.
- **Exit:** two consecutive rounds each under 3 %, or matmuls > 65 % of FLOPs and non-matmul ops < 15 % of device
  time, or 6 rounds.
- **Commits:** one per accepted change, unsigned, one lowercase line, model paths only (PLAN.md section 7).

Headline: `forward`, `paper`, `accuracy`. Guards: `long_context-accuracy`, `multivariate-accuracy`, `paper-performance`.

| round | candidate | paper wall ms | paper device ms | vs best | guards (device ms) | PCC (norm / final) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline: auto 1D matmul configs, SwiGLU as gate + up + SiLU + MUL, residual adds separate, rows padded 66 -> 96 per row, read-out after layer 23; fp32 matmul accumulation, `embed` at HiFi4 | 1201.0 | 1056.4 | — | long 95.2, multi 2003.7, perf 1044.5 | 0.999998 / 0.999999 (long 0.99987, multi 0.999997, perf 0.999975) | baseline | iter_00/summary.md, iter_00/bench.json (job 420), PCC suite 44/44 (job 457) |
