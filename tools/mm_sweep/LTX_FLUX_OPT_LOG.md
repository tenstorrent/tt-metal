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

### [2] 128x6144x768 (Mt4, narrow-N Nt24) — CLOSED: geometry/quantization limit
Selected out of rank order (highest wall/ideal of any meaningful shape = best genuine-win candidate).
- **Baseline** (config=None, mask 0): auto (1,12,1,2,1), us_med 30.3, ideal 21.9, excess 8.4µs, wall/ideal
  1.38, 370 GB/s, per-RISC B/N/T 30.2/29.5/29.7 (all cobound ~30µs), **core_spread 73%**, PCC 1.0 (corpus).
- **Sweep** (104 configs, exhaustive; ltxflux_sweep_128x6144x768.json): best (1,6,2,4,3 Sm2)=30.02µs vs
  auto 30.44µs => **1.4% headroom, below the 2-3% gate**. Top cluster (30.0-30.7) within noise. => picker
  effectively optimal (the lookup table already encodes the sweep winner for LTX/FLUX shapes by construction).
- **Ablation** (auto; ablate_128x6144x768.log): NO_REDUCE −12.3% (reduction ~3.7µs, lossy); FULL_IN0_WAIT
  +9.7%, BARRIER_DRAIN +4.9% (both adopted, active); NO_COALESCE +0.4% (narrow-N, tiny reads).
- **Hypotheses**: H1 tree-reduction to cut the split-K tail / 73% core-spread — **FALSIFIED by the sweep's own
  Pk6 control**: halving chain depth (Pk12->Pk6) gains only 1.4%, so the reduction cost is per-core
  reduce-add WORK + CB traffic, not chain depth => a tree (same work, shallower) cannot recover it. Confirms
  the prior reduction-tree rejection mechanism holds for this class too. H2 picker M-split (Sm2) — real
  direction but sub-gate (1.4%), not adopted.
- **Decision: CLOSED, geometry limit.** Narrow-N forces deep split-K for core utilization; reduction work is
  inherent. No lossless config or kernel lever. Speedup 0% (kept nothing).
- Artifacts: ltxflux_sweep_128x6144x768.json, ablate_128x6144x768.log.

### Structural finding (applies to whole corpus)
The Picker-v3 LOOKUP TABLE already stores the hand-measured sweep-winning config for every real LTX/FLUX
shape (trained to 100% on exactly these 20). => config=None already selects the sweep optimum, so **picker
headroom is ~0 by construction** on this corpus. All remaining gains must be KERNEL/dataflow improvements
(which then trigger a re-sweep + table refresh). Per-shape work therefore centers on the ablation: is there
an IDLE-RISC recoverable stall (kernel opportunity) or is the shape RISC-cobound/geometry-bound (floor)?

### [3] 64x4608x6144 (Mt2, wide-N Nt192, shallow-K Kt144) — KEPT: picker win −2.8%
Rank #2 by excess (9.7µs). First WIN. Invalidates the "table==sweep-winner for all shapes" assumption:
this shape's table entry was suboptimal.
- **Baseline** (config=None, mask 0): auto (1,6,1,1,8), us_med ~122.4, ideal 113.3, wall/ideal 1.09, 472
  GB/s, all-RISC cobound (riscImb 1%), core_spread 9%.
- **Sweep** (360 configs, broad-nsb + full-width; ltxflux_sweep_64x4608x6144.json): best (2,3,2,2,6)=118.70µs,
  top cluster all Ns2/Pk3 (118.7–119.6, 485–489 GB/s) => 3.3% headroom. => PICKER problem (not kernel).
- **A/B** (3 interleaved paired relaunches, auto vs 3 leading): auto 122.36 [121.9–122.5] vs
  (2,3,2,2,6)/(2,3,1,2,3)/(2,3,1,1,6) all 118.9–119.0 [cleanly separated] = **−2.8% stable**. Direction:
  Ns2/Pk3 (N-split + shallower split-K) beats auto Ns1/Pk6.
- **Fix**: lookup-table entry {{2,144,192}} {6,1,1,1,8} -> {3,2,1,2,3} (Ns2,Pk3,Sm1,kb2,nsb3), + python
  mirror picker_table.py. Chose the clean Sm=1 winner (tied-best, no M-split machinery).
- **Production verify** (config=None, mask 0, real op, 3 relaunches): **119.04µs [119.0–119.2]** vs old
  122.4 = **−2.8%**, 489 GB/s. Picker now returns the new config.
- **Regression**: unit 111/111, corpus 60/60 (config=None PCC fresh+cached, no other shape moved). Surgical
  keyed entry => zero effect on other shapes by construction.
- **Decision: KEPT, −2.8%.** Commit: this branch. Artifacts: ltxflux_sweep_64x4608x6144.json.
- **LESSON**: the LTX/FLUX lookup table is NOT uniformly optimal — sweep each shape rather than assume.

### [4] 128x6144x4608 (Mt4, Nt144) — CLOSED: practical floor
- Baseline: auto (1,12,1,2,1), us_med 125.6, ideal 116.0, wall/ideal 1.08, 474 GB/s, all-RISC cobound
  (riscImb 1%), core_spread 16%.
- Sweep (524 configs; ltxflux_sweep_128x6144x4608.json): best (1,6,1,2,3)=125.04µs vs auto 125.16µs =>
  **0.1% headroom**. Picker-optimal. Top cluster all 125.0–127.0 (468–475 GB/s).
- Classification: practical floor — 474 GB/s (near the ~500 read ceiling), cobound, no idle-RISC stall
  (triage riscImb 1%). No picker or kernel opportunity. Speedup 0% (kept nothing).
- Artifacts: ltxflux_sweep_128x6144x4608.json.

### [5] 128x6144x2304 (Mt4, Nt72) — CLOSED: near-optimal (1.5% sub-gate)
- Baseline: auto (1,12,1,2,1), us_med 68.6, ideal 59.5, wall/ideal 1.15, 444 GB/s, cobound (riscImb 1%),
  core_spread 28%.
- Sweep (301 configs; ltxflux_sweep_128x6144x2304.json): best (1,6,1,2,3)=66.72µs vs auto 67.72µs =>
  **1.5% headroom, below the 2-3% gate**. Direction = shallower split-K (Pk6 vs Pk12), same as the
  systematic Mt4/K6144 pattern. Not adopted (sub-gate).
- Classification: geometry/near-floor. Speedup 0% (kept nothing). Artifacts: ltxflux_sweep_128x6144x2304.json.
- PATTERN: Mt4 K=6144 shapes (768/2304/4608) all auto=Pk12 with Pk6 ~1-1.5% better — systematic sub-gate
  picker bias toward over-splitting K; not individually adoptable.

### [6] 128x2304x6144 (Mt4, wide-N Nt192, shallow-K Kt72) — CLOSED: practical floor
- Baseline: auto (2,3,1,1,6), us_med 68.4, ideal 59.5, wall/ideal 1.15, 446 GB/s, cobound (riscImb 2%,
  NCRISC-crit), core_spread 20%.
- Sweep (232 configs; ltxflux_sweep_128x2304x6144.json): best (2,3,2,1,6)=68.06µs vs auto 68.23 =>
  **0.2% headroom**. Picker already uses the good Ns2/Pk3 direction (contrast 64x4608x6144 which did not).
- Classification: practical floor/geometry. Speedup 0%. Artifacts: ltxflux_sweep_128x2304x6144.json.

### [7] 64x15360x1536 (Mt2, deep-K W5, Nt48) — KEPT: picker win −2.3%
- Baseline: auto (1,12,1,1,3), us_med 104.1, ideal 96.4, wall/ideal 1.08, 474 GB/s, cobound.
- Sweep (260 configs; ltxflux_sweep_64x15360x1536.json): best (1,6,1,2,3)=101.44µs vs auto 103.54 =>
  2.1% headroom. Direction = shallower split-K + kb2 (Pk6/kb2 vs Pk12/kb1).
- A/B (3 interleaved paired): auto 103.67 [103.5–104.1] vs (1,6,1,2,3) 101.32 [101.3–101.3] =
  **−2.3% cleanly separated**; (1,3,1,4,6) −1.2% (worse).
- Fix: {{2,480,48}} {12,1,1,1,3} -> {6,1,1,2,3} + python mirror.
- Production verify (config=None, mask 0): 101.85µs [runs the new cfg] = −1.8% single-pass / −2.3% A/B, 486 GB/s.
- Regression: unit 111/111 + corpus 60/60 (config=None PCC fresh+cached; no other shape moved).
- Decision: KEPT, −2.3% (A/B; low-boundary of gate but stable + cleanly separated + zero-regression keyed fix).
- Artifacts: ltxflux_sweep_64x15360x1536.json.
