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

### [1] 128x15360x768 (Mt4, deep-K W5, narrow-N) — CLOSED: practical floor
- **Baseline** (config=None, mask 0): auto cfg (1,6,1,2,3), us_med 66.0, ideal 54.1, excess 11.85µs,
  wall/ideal 1.22, 420 GB/s, per-RISC B/N/T = 66.1/64.6/65.2 (all saturated ~65µs; BRISC/in1-read
  marginally critical), core_spread 41%, sched_over_valid 1.0, PCC 1.0000 (fresh+cached, corpus smoke).
- **Sweep** (98 feasible configs, exhaustive; ltxflux_sweep_128x15360x768.json): **AUTO is the best config
  — 0.0% headroom.** Top: (1,6,1,2,3)=65.4µs, (1,6,2,2,3)=69.4, (1,12,1,1,3)=70.3, (1,8,1,2,3)=77.3. Sm>1
  and all other Pk/kb/nsb factorizations are slower. => NOT a picker problem.
- **Ablation** (auto cfg; ablate_128x15360x768.log): NO_REDUCE −4.0% (reduction ~2.6µs, lossy);
  FULL_IN0_WAIT +29.5% (progressive-wait, adopted, is the dominant captured win); NO_COALESCE +2.6%,
  BARRIER_DRAIN +1.6%, FWD_FLUSH_FIRST +0.5% — all adopted optimizations active and helping. Even with
  ALL reduction removed (lossy) 8.3µs excess remains => residual is in1-read BW efficiency (420 vs ~500
  GB/s ceiling) + reduction-root asymmetry, not a discrete removable stall.
- **Hypotheses**: H1 reduction restructure to cut the 41% split-K root asymmetry — FORECLOSED (NO_REDUCE
  ceiling 4% lossy; reduction-tree already measured sub-bar for this class). H2 finer in0/in1 delivery
  granularity — FORECLOSED (in0-chunk C1 measured NEUTRAL on this exact shape). H3 higher in1-read BW — no
  mechanism (all 3 RISCs saturated, coalescing already adopted; no idle engine to recover).
- **Decision: CLOSED, practical floor.** No lossless kernel/picker opportunity. Speedup 0% (kept nothing).
- Artifacts: ltxflux_sweep_128x15360x768.json, ablate_128x15360x768.log. Commit: (harness) this branch.
