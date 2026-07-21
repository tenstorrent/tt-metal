# LTX/FLUX Mt≤8 cumulative optimization campaign

Branch: `cglagovich/regime-a-ltxflux-opt` (based on single-chip head `323ae42b161`; AGMM excluded).
Production path only: `config=None`, mask 0, BF16 in/out, FP32 acc, HiFi2. Random-operand PCC≥0.999 +
fresh/cached correctness required for every adopted change. Skip-work ablations are causal evidence only.

Harness: `ltxflux_opt.py` (feasible-space enumeration + per-config sweep via the real production kernel,
diag entry mask 0, forced config). Regression gates: `test_regime_a_matmul{,_corpus}.py` (111 + 60),
parity gtest, and `regime_a_current_perf.py` full-corpus perf. Perf baseline: `regime_a_current_perf.json`.

## Corpus: real LTX/FLUX Mt≤8 = 19 shapes (M∈{32,64,128}; (512,6144,1536) is Mt=16 → separate/diagnostic)

## Frozen baseline ranking (production config=None, mask 0; from regime_a_current_perf.json, 2026-07-21)
Ranked by absolute excess over the DRAM floor (excess_us = us_med − ideal_us).

| rank | shape | Mt | excess µs | wall/ideal | GB/s | us_med | ideal | auto cfg (Ns,Pk,Sm,kb,nsb) |
|---|---|---|---|---|---|---|---|---|
| 1 | 128x15360x768 | 4 | 11.9 | 1.22 | 420 | 66.0 | 54.1 | (1,6,1,2,3) |
| 2 | 64x4608x6144 | 2 | 9.7 | 1.09 | 472 | 122.9 | 113.3 | (1,6,1,1,8) |
| 3 | 128x6144x4608 | 4 | 9.7 | 1.08 | 473 | 125.6 | 116.0 | (1,12,1,2,1) |
| 4 | 128x6144x2304 | 4 | 9.0 | 1.15 | 444 | 68.6 | 59.5 | (1,12,1,2,1) |
| 5 | 128x2304x6144 | 4 | 8.8 | 1.15 | 446 | 68.4 | 59.5 | (2,3,1,1,6) |
| 6 | 128x6144x768 | 4 | 8.4 | 1.38 | 370 | 30.3 | 21.9 | (1,12,1,2,1) |
| 7 | 64x15360x1536 | 2 | 7.7 | 1.08 | 474 | 104.1 | 96.4 | (1,12,1,1,3) |
| 8 | 64x6144x9216 | 2 | 7.1 | 1.03 | 496 | 232.1 | 225.0 | (1,6,1,4,2) |
| 9 | 32x6144x9216 | 1 | 5.7 | 1.03 | 499 | 228.8 | 223.1 | (1,3,1,4,6) |
| 10 | 64x6144x4608 | 2 | 5.6 | 1.05 | 488 | 118.9 | 113.3 | (1,6,1,4,2) |
| 11 | 64x6144x1536 | 2 | 5.6 | 1.14 | 448 | 44.4 | 38.8 | (1,12,1,2,1) |
| 12 | 32x6144x2304 | 1 | 4.9 | 1.09 | 471 | 61.3 | 56.4 | (1,4,1,2,9) |
| 13 | 32x6144x6144 | 1 | 4.5 | 1.03 | 497 | 153.5 | 149.0 | (1,6,1,4,2) |
| 14 | 32x6144x3072 | 1 | 3.7 | 1.05 | 488 | 78.6 | 74.9 | (1,3,1,4,6) |
| 15 | 32x6144x1536 | 1 | 3.5 | 1.09 | 469 | 41.3 | 37.8 | (1,6,1,4,2) |
| 16 | 32x2048x512 | 1 | 3.0 | 1.68 | 305 | 7.4 | 4.4 | (2,4,1,2,1) |
| 17 | 32x2048x2048 | 1 | 2.5 | 1.15 | 447 | 19.4 | 16.9 | (2,2,1,4,4) |
| 18 | 32x2048x1536 | 1 | 2.3 | 1.18 | 434 | 15.0 | 12.7 | (2,2,1,4,3) |
| 19 | 32x256x6144 | 1 | 1.7 | 1.25 | 411 | 8.6 | 6.9 | (3,1,1,1,8) |

Notes: 32x2048x512 has the worst ratio (1.68) but tiny absolute excess (3.0µs) — low priority per ranking
rule. Deep-K narrow-N Mt=4 (128x15360x768) is the top absolute opportunity. Shapes 8–10,13 are already
near the DRAM floor (wall/ideal ≤1.05, ≥488 GB/s) — likely "practical floor" closes.

---
## Iteration log
(entries appended per shape below)
