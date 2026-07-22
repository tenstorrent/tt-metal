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

### [8] 64x6144x1536 (Mt2, Nt48) — KEPT: picker win −3.5%
- Baseline: auto (1,12,1,2,1), us_med 44.4, ideal 38.8, wall/ideal 1.14, 448 GB/s.
- Sweep (229 configs; ltxflux_sweep_64x6144x1536.json): best (1,3,1,8,2)=42.87µs vs auto 44.35 => 3.4%.
  Direction = much shallower split-K + big kb (Pk3/kb8 vs Pk12/kb2).
- A/B (3 interleaved): auto 44.34 [44.25–44.41] vs (1,3,1,8,2) 42.77 [42.71–42.78] = **−3.5% cleanly
  separated**; (1,6,1,4,2) −2.7%.
- Fix: {{2,192,48}} {12,1,1,2,1} -> {3,1,1,8,2} + mirror. KEPT.
- Artifacts: ltxflux_sweep_64x6144x1536.json.

### [9] 32x6144x2304 (Mt1, Nt72) — KEPT: picker win −3.8%
- Baseline: auto (1,4,1,2,9), us_med 61.3, ideal 56.4, wall/ideal 1.09, 471 GB/s.
- Sweep (133 configs; ltxflux_sweep_32x6144x2304.json): best (1,3,1,4,5)=58.72µs vs auto 60.86 => 3.6%.
  Direction = Pk3/kb4/nsb5 vs Pk4/kb2/nsb9 (picker picked a poor nsb=9 too).
- A/B (3 interleaved): auto 60.99 [60.8–61.1] vs (1,3,1,4,5) 58.69 [58.6–58.75] = **−3.8% cleanly
  separated**; (1,3,1,8,2) −2.2%.
- Fix: {{1,192,72}} {4,1,1,2,9} -> {3,1,1,4,5} + mirror. KEPT.
- Artifacts: ltxflux_sweep_32x6144x2304.json.

### [10] 32x6144x1536 (Mt1, Nt48) — CLOSED: picker optimal (0.3%)
- Baseline: auto (1,6,1,4,2), 469 GB/s, wall/ideal 1.09. Sweep (125 configs): best (1,3,1,4,6)=40.46µs vs
  auto 40.60 => 0.3%. Auto already low-Pk (Pk6) => no over-split; picker correct. Speedup 0%.
- Artifacts: ltxflux_sweep_32x6144x1536.json.

### [11-14] Small shapes — CLOSED: picker-optimal (0.0% headroom each)
- 32x2048x512 (auto (2,4,1,2,1), wall/ideal 1.68): sweep 24 cfgs, best=auto, 0.0%. High ratio is pure
  geometry (Kt64 shallow, tiny output => overhead-dominated), not config-recoverable. Practical floor.
- 32x2048x2048 (auto (2,2,1,4,4)): 101 cfgs, 0.0%. 32x2048x1536 (auto (2,2,1,4,3)): 75 cfgs, 0.0%.
- 32x256x6144 (auto (3,1,1,1,8), Kt8 ultra-shallow): 18 cfgs, 0.0%. wall/ideal 1.25 = geometry.
- All auto already low-Pk => picker correct. Kept nothing. Artifacts: ltxflux_sweep_32x2048x{512,2048,1536}.json,
  ltxflux_sweep_32x256x6144.json.

### [15-19] Near-DRAM-ceiling group — CLOSED: practical floor
Five shapes at 488–499 GB/s (95–97% of 512 peak, at/near the ~500–511 read ceiling), all with low-Pk autos
and cobound (triage riscImb 0–1%). Confirmed picker-optimal on the two weakest-BW representatives by
exhaustive sweep; the higher-BW members follow by the same low-Pk-auto=optimal pattern.
- 64x6144x4608 (auto (1,6,1,4,2), 489 GB/s): 480 cfgs swept, 0.2% headroom. FLOOR.
- 32x6144x3072 (auto (1,3,1,4,6), 494 GB/s): 229 cfgs swept, 0.3% headroom. FLOOR.
- 64x6144x9216 (auto (1,6,1,4,2), 496 GB/s, wall/ideal 1.03), 32x6144x9216 (auto (1,3,1,4,6), 499 GB/s,
  1.03), 32x6144x6144 (auto (1,6,1,4,2), 497 GB/s, 1.03): group-closed — highest BW% in the corpus, low-Pk
  autos, cobound; no config or kernel headroom. Kept nothing.
- Artifacts: ltxflux_sweep_64x6144x4608.json, ltxflux_sweep_32x6144x3072.json.

## CAMPAIGN SUMMARY (all 19 LTX/FLUX Mt≤8 shapes examined)
**4 KEPT picker wins** (all stable, A/B-confirmed, gate-passing, zero-regression keyed lookup-table fixes):
| shape | old cfg | new cfg | A/B | config=None |
|---|---|---|---|---|
| 64x4608x6144 | (1,6,1,1,8) | (2,3,1,2,3) | −2.8% | −2.8% |
| 64x15360x1536 | (1,12,1,1,3) | (1,6,1,2,3) | −2.3% | −1.8% |
| 64x6144x1536 | (1,12,1,2,1) | (1,3,1,8,2) | −3.5% | −3.0% |
| 32x6144x2304 | (1,4,1,2,9) | (1,3,1,4,5) | −3.8% | −2.1% |

**15 CLOSED with evidence** (exhaustive sweep ~0 headroom + ablation/triage): practical floor or geometry.
**Root cause of all wins**: the picker's cost-model fallback OVER-SPLITS K (high Pk) on some Mt1/Mt2 K=6144/
15360 shapes; a shallower Pk3–6 with larger kb (fewer, deeper k-slices) delivers the same parallelism with
less split-K reduction overhead. Shapes whose auto already used low Pk were all picker-optimal.
**No kernel change was warranted**: the kernel is RISC-cobound corpus-wide (0–2% RISC imbalance, fully
overlapped); no idle-RISC recoverable stall exists. Reduction-tree / in0-chunk / in1-BW levers stayed
foreclosed (fresh ablations: NO_REDUCE lossy, all adopted opts active; Pk6 controls show reduction cost is
work- not depth-bound). All 4 gains are picker-only.

## ============ DEEP KERNEL INVESTIGATION (new goal) ============
### [DEEP-1] 256x2048x1024 (Mt8, Kt64 shallow-K, Nt32, W1) — DIAGNOSED; hypotheses ranked, pending review
**Baseline** (config=None, mask 0, resident inputs): auto (1,4,2,2,4)=Ns1/Pk4/Sm2/kb2/nsb4, 64 cores, W1.
us_med 22.33 [22.13/22.33/22.50], DRAM ideal 11.26, **wall/ideal 1.98**, delivered 258 GB/s, core_spread
59%, per-RISC B/N/T 22.5/20.6/21.1 (BRISC/in1-read critical), PCC 1.0000 (fresh+cached), ai 186.

**Full sweep** (planner-complete: Pk1-12, Ns1-6, Sm1-8, kb{1,2,4,8}, all nsb): enumerated 4224, feasible/
measured 264, rejected 3960, failed 1. **BEST = AUTO, 0.0% headroom** (2nd (1,4,3,2,4) Sm3 +3.5%). =>
NOT a picker problem; kernel/dataflow-limited. Artifact: ltxflux_sweep_256x2048x1024.json.

**Causal timing decomposition** (DIAG_ZONES=1<<4 compile-gated DeviceZoneScopedN; perturbation = DIAG_ZONES
22.81us vs mask-0 22.33 = **+2.1%**, small). Per-zone med across cores [min..max], spread=core imbalance:
| zone (role) | med us | min | max | spread% |
|---|---|---|---|---|
| Z_IN1READ (reader/BRISC) | 12.4 | 10.2 | 16.8 | 64% |
| Z_RING (writer in0 all-gather) | 11.2 | 9.0 | 13.7 | 52% |
| Z_PHASE2 (writer reduce+output) | 6.4 | 2.6 | 13.0 | **392%** |
Writer NCRISC = Z_RING(11.2)+Z_PHASE2(6.4) ~= 17.5us; reader BRISC = Z_IN1READ(12.4). Wall 22.8.
Two large, previously-hidden costs: (a) **in0 ring ~11us** — ~half the wall for only ~1MB of in0 (serial
8-hop forwarding exposed at shallow-K/W1); (b) **Z_PHASE2 392% root-vs-leaf imbalance** — the is_top core
serially does receive-4-partials + accumulate + write-all-output while leaves idle.

**Controlled ablations** (auto cfg; ceilings, LOSSY unless noted):
| ablation | wall | delta | reads as |
|---|---|---|---|
| baseline | 22.43 | — | |
| NO_REDUCE | 18.68 | **-16.7%** | split-K reduce+root-output tail = ~3.75us (matches Z_PHASE2 root) |
| FULL_IN0_WAIT | 26.08 | +16.3% | progressive-wait (adopted) already hides much in0-ring latency |
| NO_COALESCE | 23.07 | +2.9% | in1 coalescing adopted, active |
| FWD_FLUSH_FIRST | 22.92 | +2.2% | in1 fwd-order adopted |
| PLACE_READERS_FIRST | 22.91 | +2.1% | current placement adopted |
| BARRIER_DRAIN | 22.54 | +0.5% | pipelined drain ~ neutral here |

**Ranked hypotheses** (predicted zone + ceiling):
- **H1 (lead): reduction/output-tail overlap on the split-K root.** Obs: Z_PHASE2 392% spread, root ~13us
  vs leaf 2.6us; NO_REDUCE ceiling -16.7% (lossy). Mechanism: is_top serializes receive->accumulate->
  output-DRAM-write; overlap the per-block output write with the next partial's receive/add (pipeline the
  root), and/or start writing reduced blocks as they complete rather than after the whole chain. Predicted:
  shrink Z_PHASE2 root; realizable ~3-6% (fraction of the 16.7% lossy ceiling). Negative control: a Pk=1
  shape (no reduction) must be unaffected. Falsifier: if root time is dominated by the DRAM write itself
  (not the serialize), overlap won't help. Lossless.
- **H2: in0-ring exposure for shallow-K/W1.** Obs: Z_RING ~11us (~half wall) for ~1MB in0. Mechanism: the
  serial G=8 forwarding chain latency is exposed when per-shard payload is tiny (W1) so BW can't amortize the
  hop latency. Prior scatter/exchange REJECTED — but on DEEP-K shapes where the ring was hidden; shallow-K/W1
  with the ring exposed is a NEW regime (prior rejection mechanism = "ring hidden" no longer holds). Needs a
  skip-in0 ablation (removed in cleanup) re-added to bound the ceiling before prototyping. Predicted: earlier
  publication / shorter effective chain shrinks Z_RING.
- H3 (minor): in1-read efficiency (Z_IN1READ 12.4us, 64% spread) — shallow per-core K reads; coalescing
  already adopted (+2.9%). Lower priority.

**Status**: baseline + full sweep + zone decomposition + ablation ceilings + ranked hypotheses COMPLETE.
Prototyping H1/H2 is the next step. Instrumentation (DIAG_ZONES) is committed + mask-0 byte-identical
(regression gate). **PAUSING for review per goal before prototyping** (esp. H2 reopens a previously-rejected
direction — worth a steer). Artifacts: ltxflux_sweep_256x2048x1024.json, zone_parse.py, this log.

### [DEEP-1 REV2] 256x2048x1024 — corrected timeline (per review feedback)
Fixes: harness reads "risc" (not per_risc_us); failed sweep cfg (2,1,6,1,2) classified = TRANSIENT (reruns ok,
77us non-competitive); added compute-side zones + raw absolute-timestamp artifact (zone_raw_256x2048x1024.json,
800 core-zone series). Perturbation with compute zones = +0.6%.

**Compute-side decomposition (TRISC per-core; the missing piece).** Regular start/end zones around the k-loop
waits (SumN accumulators do NOT flush for the compute kernel; K_num_blocks small so per-block markers cheap):
| TRISC | median us | max us |
|---|---|---|
| matmul (residual) | 9.1 | 14.4 |
| in0-wait (ring exposed to compute) | 4.9 | 8.4 |
| in1-wait | 1.4 | 7.3 |
=> the coarse Z_RING (11us) is NOT all exposed: only ~4.9us of in0-ring latency is exposed as COMPUTE stall;
in1 read is largely hidden (1.4us). Whole-phase duration != exposed headroom (confirmed).

**Per-role timeline + wall accounting (from absolute timestamps).** Roles: 32 readers (Z_IN1READ) / 32 slaves;
27 split-K roots / 37 leaves. The WALL (~22.3us) is gated by ROOT cores: writer(NCRISC) does Z_RING(~11us)
THEN Z_PHASE2 reduce+output(~9us) SEQUENTIALLY. Root Z_PHASE2 med 9.0us vs leaf 4.8us (+4.2us root tail).
Reader Z_IN1READ (~13us) OVERLAPS the writer and is not the tail. Accounted wall ~= ring(11) + root
reduce/output(9) + startup ~= 22us.

**Refined hypotheses (ceilings need the NCRISC sub-zone breakdown — next increment):**
- H-A: **reduction/output root tail** extends the wall ~4-5us beyond compute (root Z_PHASE2 9us, +4.2us vs
  leaf; root ends the wall). NOT "add a per-block output pipeline" (already exists). Need Z_PHASE2 sub-zones
  (partial-recv-wait / out_cb-wait / output-issue+flush) to locate the recoverable sub-component.
- H-B: **in0-ring on the critical path AND exposed** (~4.9us compute in0-wait; ring is the sequential first
  half of the root path). Reopen in0 ONLY with a NOT-already-tested mechanism (scatter/exchange/repl/chunk
  all rejected ON THIS SHAPE). Need ring sub-zones (inject / recv-wait / forward) to bound the recoverable.

**Status: NOT prototyping.** Next: NCRISC ring + reduction sub-zones -> numerical ceilings + precise
experiments, then report. Artifacts: zone_raw_256x2048x1024.json, zone_parse.py, ltxflux_sweep_256x2048x1024.json.

### [DEEP-1 REV3] 256x2048x1024 — reduction sub-zone breakdown (numerical ceilings)
Added writer PHASE2 sub-zones (DIAG_ZONES): Z_P2_RECVWAIT (wait upstream partial), Z_P2_OUTWAIT (wait
compute block), Z_P2_OUTWRITE (root DRAM output issue+flush). ROOT cores (16, is_top), per-iter med:
| root sub-zone | med us | max us |
|---|---|---|
| Z_PHASE2 (total) | 9.38 | 12.51 |
| **Z_P2_RECVWAIT** | **7.1** | 10.46 |
| Z_P2_OUTWAIT | 0.76 | 0.76 |
| Z_P2_OUTWRITE | 0.89 | 1.43 |
=> **The root's reduction tail is WAIT-bound (7.1us waiting for the Pk=4 chain), NOT output-write-bound
(0.9us).** This REFUTES the earlier H1 (overlap output write — the write is only 0.9us). Also DIFFERENT from
the deep-K shapes where reduction was WORK-bound (there the prior reduction-tree rejection mechanism applied);
here it is chain-WAIT/latency-bound, a regime the prior rejection did NOT cover.

**Accounted wall model (root core, ~22.5us wall):** ring(~11us, overlaps compute matmul+in0-wait) THEN
Z_PHASE2 = RECVWAIT(7.1) + OUTWAIT(0.8) + OUTWRITE(0.9) ~= 9us. Compute (TRISC) ~17us (matmul 9 + in0-wait
4.9 + in1-wait 1.4 + pack/oh ~1.7). Wall ~= compute-end + reduction-chain-drain, with RECVWAIT the dominant
post-compute critical-path component.

**Ranked realizable hypotheses (numerical ceilings + precise experiments):**
- **H-A (LEAD): split-K reduction-chain RECVWAIT = 7.1us on the root** (dominant Z_PHASE2 component, on the
  post-compute critical path). Ceiling: up to ~7us if the root did not serialize on chain propagation.
  EXPERIMENT A1 (diagnostic first, no prod change): instrument RECVWAIT by chain position (bottom/middle/
  root) — if RECVWAIT grows with position => propagation-latency (a shallower/tree reduction or fewer hops
  helps); if uniform+large => upstream compute-imbalance (balance/faster-finish helps); if forward-BW =>
  large out_blk forwards (coalesce/smaller granularity). EXPERIMENT A2 (only if A1 shows propagation): A/B a
  2-level tree for Pk=4 (2 hops vs 3) — reopens reduction-tree with NEW evidence (wait-bound, not work-bound;
  prior rejection was work-bound on deep-K). Predicted zone: shrink Z_P2_RECVWAIT.
- **H-B: in0-ring exposed to compute = 4.9us** (TRISC in0-wait). Ceiling ~4.9us. EXPERIMENT B1: ring
  sub-zones (inject / recv-wait / forward) to locate the exposed latency; only propose an in0 change if it is
  a mechanism NOT already tested on this shape (scatter/exchange/repl/chunk all rejected here).

**Status: diagnosis complete with numerical ceilings + experiments. NO prototype.** Lead = H-A (A1 diagnostic
next). Artifacts: zone_raw_256x2048x1024.json, LTX_FLUX_OPT_LOG.md.

### [DEEP-1 REV4] 256x2048x1024 — reduction realizability (NO_REDUCE overlap check; RECVWAIT-fraction red herring)
A1 (RECVWAIT by chain position): mids(pos1-2) med 4.2us vs roots(pos3) 7.1us, ~2.4us/hop => PROPAGATION-
latency-bound (not compute-imbalance). BUT the recoverable amount != RECVWAIT (overlap): NO_REDUCE realizable
ceiling A/B (3x):
| shape | Pk | AI/BW | base->NO_REDUCE | reduction ceiling |
|---|---|---|---|---|
| 256x2048x1024 | 4 | shallow-K, 258 GB/s (NOT DRAM-bound) | 22.3->18.8 | **16% (3.5us) — EXPOSED** |
| 32x6144x1536 | 6 | deep-K, ~490 GB/s (DRAM-bound) | 40.5->39.4 | 3% (1.1us) — overlapped |
| 32x6144x6144 | 6 | deep-K, DRAM-bound | 152.8->151.6 | 1% (1.3us) — overlapped |
**RED HERRING CORRECTED:** deep-K root RECVWAIT is 80-91% of wall but OVERLAPS the in1-read floor (those
shapes are DRAM-read-bound; earlier "floor" close was correct). Reduction is only recoverable on SHALLOW-K
exposed shapes (256x2048x1024, 32x2048x512). A tree cuts propagation DEPTH (Pk4: 3->2 hops) not total adds,
so realizable ~1-1.5us (~5-7%) on 256x2048x1024, ONLY on Pk>=3 shallow-K (Pk2 chain is already depth 1;
deep-K DRAM-bound; Pk1 no reduction). Net: the reduction tree is a shallow-K-Pk>=3-only lever, ~5% on ~2
corpus shapes, and reopens a previously-rejected direction — worth a gated prototype but modest+risky.

**256x2048x1024 exposed-lever summary (wall 22.3us, ideal 11.3us):** matmul 9us (compute floor, lossless-
irreducible) + in0-ring 4.9us exposed (foreclosed: scatter/exchange/repl/chunk all rejected on this shape) +
reduction 3.5us exposed (tree ~1-1.5us realizable) + in1-wait 1.4us. Realizable headroom ~= tree only (~5%).

### [DEEP-1 FINAL] 256x2048x1024 — CLOSED: no high-value lossless lever (tree deferred with evidence)
Config-level realizability test of the reduction-depth hypothesis: shorter-chain configs at the SAME core
count are much SLOWER, not faster:
| cfg | Pk/Sm | cores | chain depth | wall | vs auto |
|---|---|---|---|---|---|
| (1,4,2,2,4) AUTO | 4/2 | 64 | 3 | 22.08 | — |
| (1,2,4,4,4) | 2/4 | 64 | 1 | 28.40 | **+29%** |
| (1,2,2,4,4) | 2/2 | 32 | 1 | 25.84 | +17% |
All low-Pk (<=2) configs are +17..+30%. So config-level chain-shortening is net-negative (deeper per-core K
+ more M-split forwarding outweigh the shorter chain). A topology-only tree (keep Pk4/Sm2, reduce reduction
DEPTH 3->2) MIGHT realize ~1.2us (~5%) per A1+NO_REDUCE, but: (a) absolute is negligible (1.2us on the worst
shape; ranking guidance de-prioritizes negligible absolutes), (b) reopens a previously-REJECTED mechanism,
(c) requires a high-risk multi-child-receive reduction rewrite (CorePlan + runtime args + writer protocol +
semaphore counting; deadlock-prone), (d) the only config-level realizability signal is NEGATIVE (+29%).
=> **DECISION: defer the tree (documented design + ceiling below); CLOSE 256x2048x1024.**

**Accounted wall (22.3us, ideal 11.3us):** matmul 9us (compute floor) + in0-ring exposed 4.9us (FORECLOSED:
scatter/exchange/repl/chunk/direct-read all rejected on this shape) + reduction exposed 3.5us (tree ~1.2us
realizable, deferred) + in1-wait 1.4us. No lossless lever with a favorable risk/reward. The shape is
overhead-bound by split-K + ring + M-split delivery on tiny per-core work (low AI after 64-way split).
**Deferred tree design (for future, if a higher-Pk shape makes it worthwhile):** balanced binary reduction,
internal node fan-in 2 (wait num_children partials, add, forward to parent); build_plan assigns tree parent/
children by redpos; ceiling grows with Pk (Pk4:~5%, Pk12:~15% on EXPOSED shapes only — deep-K stays
DRAM-bound/overlapped so no benefit there). Speedup kept: 0%.
