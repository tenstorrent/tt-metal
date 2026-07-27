# Regime-A picker generalization — generator-driven campaign report

**Scope.** Measurement-driven evaluation of the `regime_a_matmul` production picker across a broad
boundary-aware corpus, using the theory-guided candidate generator
(`tools/mm_sweep/regime_a_candidate_generator.py`) to prune the search. Dataset + offline analysis only;
**no C++ picker change was made.** Single Blackhole p150b, 1.35 GHz, 110-core grid.

## 1. Method

- **Candidate selection.** Per shape, `select_candidates(M,K,N, budget=96, audit=8, include=[prod])`
  from the generator: an ensemble of physical cost models (DRAM / compute / latency / reduction /
  m-forward / streaming) plus factorization- and class+cores-diversity, always including the real
  production-picker config, plus 8 deterministic **audit** samples drawn from the *pruned* feasible
  space to test the pruning. No old exhaustive nsb lattice.
- **Persistent batch worker** (`batch_worker.py`). Imports TTNN once; opens the device once per
  12-config mini-batch; recovers per-config kernel wall from the profiler CSV by **run-host-id demux**
  (CSV flushes on close); atomic JSON checkpoint per config; PCC vs an fp32 reference computed once per
  shape. Same timed work as the isolated harness (resident inputs, 1 warmup + 8 timed iters). Verified
  within **0.62%** of the isolated one-process-per-config method on 5 configs (`verify_batch_timing.py`),
  PCC 0.99999. Throughput ≈ **0.9 s/config** (was 7.5 s isolated).
- **Supervisor** (`batch_sweep.py`). Generator → job file (carrying each candidate's geometry / model
  costs / reasons) → worker; progress-watchdog kills + `tt-smi -r` + resumes on stall. Fully resumable.
- **Corpus** (`corpus_v2.py`, 141 shapes): matched anchors Mt=1..8 × 12 (K,N) = 96 + all real FLUX/LTX
  Mt≤8 + Mt{1,4,8} boundary pairs + tail/Nt%8; train/val/holdout pre-assigned with real+tail in
  val/holdout as well as train.
- **Stability.** Winner + every candidate within 5% + top-5 remeasured with 3 sequential fresh
  worker/device relaunches (not interleaved), PCC verified. 1,050 near-winners × 3 = 3,150 reruns.

**Scale.** 13,398 initial measurements (13,374 ok / 24 no-wall) + 3,150 reruns, ~3.3 h + ~2.6 h, **0
hangs**. Winner PCC **141/141 ≥ 0.99998**; winner rerun spread median **0.73%**, max 4.0%.

## 2. Generator validation (pruning is safe)

- **Unit tests** `test_regime_a_candidate_generator.py`: 4/4 pass.
- **Historical backtest** (4 prior exhaustive sweeps): **0.00%** regret on all 4 — exact optimum retained.
- **Pruning-regret vs the 11 aborted-campaign exhaustive shapes** (`PRUNING_REGRET.md`): optimum retained
  **8/11**; regret median **0.00%**, worst **+2.36%**; within-2% for **10/11**. The 3 misses are all
  small-Mt (1/2/3); the large-Mt production regime is exact. → the generator is a safe candidate producer.
- **In-campaign pruning audit**: across all 141 shapes, an audit (pruned-space) candidate beat the best
  model-selected candidate in **0/141** shapes, and the overall optimum was an audit candidate in
  **0/141**. The pruning excluded no winners on this corpus.

## 3. Production-picker gap vs the measured optimum

| subset | n | geomean | median | worst | >3% | >5% |
|---|---|---|---|---|---|---|
| ALL | 141 | 7.8% | 4.1% | 53.5% | 75 | 63 |
| train | 83 | 9.7% | 5.8% | 53.5% | 47 | 44 |
| val | 29 | 7.6% | 4.4% | 50.4% | 17 | 13 |
| holdout | 29 | 3.1% | 1.4% | 17.2% | 11 | 6 |
| **FLUX/LTX** | 42 | **1.0%** | **0.0%** | 9.9% | 6 | 3 |

**The production picker is near-optimal on the FLUX/LTX shapes it was tuned for (median 0.0%, its lookup
table) but generalizes poorly** — geomean 7.8% and up to **53.5%** off on the broader corpus, i.e. wherever
the cost-model *fallback* is exercised.

## 4. Major outliers (investigated, not hidden)

The largest gaps cluster tightly on **deep-K, moderate-N** shapes that miss the lookup table and hit the
fallback:

| cluster | prod pick | measured optimum | gap |
|---|---|---|---|
| `*x8192x1536` (Kt=256, Nt=48), Mt=3/4/5/6/7/8 | `Ns=3,Pk=4,kb=2` | `Ns=1,Pk=8–11,kb=2–3` | 26–54% |
| `*x7168x1280` (Kt=224, Nt=40) | `Ns=1,Pk=10,kb=1` | `Ns=1,Pk=7,kb=4` or `Sm=2,kb=7` | 33–52% |
| `*x4608x3072` (Kt=144, Nt=96) | `Ns=4,Pk=3,kb=2` | `Ns=1–2,Pk=3–9,Sm≤2` | 24–36% |

Root cause (verified on `256x8192x1536`: prod `Ns=3` 128.6 µs vs opt `Ns=1,Pk=11` 87.6 µs, **−46.9%**):
**the fallback over-splits N and under-splits K on deep-K shapes.** It selects N-split (`Ns=3`) where the
optimum is a single N-band with deep split-K (`Ns=1`, high `Pk`) — the whole top-5 for these shapes is
`Ns=1`. The fallback's `readT = Kt·Nt/min(cores,24)` term saturates at 24 cores, so it cannot tell that
adding K-split parallelism past ~24 readers still helps these deep-K shapes, and it defaults to N-split.
Its `kb` choice is also often too shallow (`kb=1–2` vs optimal `kb=3–7`) on the deep-K clusters.

## 5. Conclusion for orchestrator review

- **The generator is validated** as a safe, ~100×-pruned candidate producer: 0.00% historical regret,
  0/141 in-campaign pruning misses, small-Mt worst-case 2.36% regret on the aborted exhaustive set.
- **The evidence is strong that the production *fallback* mis-generalizes** (median 5.8% on train, up to
  53% on deep-K/moderate-N non-table shapes), while the FLUX/LTX lookup table itself is fine (median 0%).
  A production-picker change is justified **for the fallback**, targeting the deep-K K-vs-N split decision
  (prefer `Ns=1` + deeper `Pk`/`kb` when `Kt` is large and `Nt` is moderate) — but that is a **separate,
  gated change**, deliberately not made here.
- **Deliverables** (this directory): `regime_a_candidate_generator.py` integration, `batch_worker.py` +
  `batch_sweep.py` (resumable harness), `corpus_v2.py` + `corpus_v2_manifest.json`, `results_v2/` (raw
  candidate results + stability reruns, one JSONL per shape), `verify_batch_timing.py`, `pruning_regret.py`
  + `PRUNING_REGRET.md`, `analyze_v2.py` + `analysis_v2_summary.json`, `CAMPAIGN_REPORT_gaps.md`, this
  report. Aborted-campaign exhaustive calibration data in `results/`.

## 6. Reproduce / resume

```bash
source /home/cglagovich/bh_env.sh && source python_env/bin/activate
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
cd $TT_METAL_HOME
python3 tools/mm_sweep/picker_gen/corpus_v2.py                      # (re)build the manifest
python3 tools/mm_sweep/picker_gen/batch_sweep.py initial            # resumable; skips measured configs
python3 tools/mm_sweep/picker_gen/batch_sweep.py rerun              # stability + PCC on near-winners
python3 tools/mm_sweep/picker_gen/batch_sweep.py status             # progress
python3 tools/mm_sweep/picker_gen/analyze_v2.py --md REPORT_gaps.md # gaps + pruning audit
python3 tools/mm_sweep/picker_gen/pruning_regret.py --md PRUNING_REGRET.md
(cd tools/mm_sweep && python3 -m pytest test_regime_a_candidate_generator.py -q)
```
