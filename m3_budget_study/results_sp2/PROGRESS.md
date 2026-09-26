# M3 prefill SP=2 follow-up — progress

Branch `vmelnykov/m3_prefill_sp2_study` from `vmelnykov/m3_prefill_budget_study` (`ba334d9`). Host bh-glx-120-b09u02.
Everything for this study is under `m3_budget_study/results_sp2/` (self-contained, download separately).
Lock: `results_sp2/.lock` (owner = session vmelnykov-2e); `run_budget.sh` refuses to run if the lock is not ours.

## Setup
- `run_budget.sh`: `BUDGET_RESULTS` (results root) and `BUDGET_LOCK` / `BUDGET_LOCK_OWNER` guard (default off).
- `budget_sweep.py` / `budget_packed.py`: `BUDGET_ANY_LAYERS=1` (S8' = layers 8–15 straddles the 4-stage carve's
  stage 0 = 0–14), `BUDGET_MEM=1` (DRAM in use after the last point).
- Added A5 (not in the doc): each full pipeline stage in isolation, W=4096, h = 0 / 139k / 549k, with DRAM report.

## Runs
| run | what | status | note |
|---|---|---|---|
| sp2_anchor_s15_w4096 | §2 sanity (layers 15–29, (2,4), W=4096, cold) | OK | **238.2 ms vs ≈241 (−1.2%)** = 31.0 chip-µs/token-layer |
| a1_d2_*, a1_s8p_* | A1 cold width (plain) | OK | D2 9.9/16.7/35.4 ms at W=2048/4096/8192 |
| a2_d2_grid, a2_s8p_grid | A2 depth, W=2048, n ∈ {256, 2048} | OK | dense: no n dependence; sparse 549k 93.1 ms / 8 layers |
| a2w_* | A2w wide depth | OK (partial at 21:19) | dense 549k: 177.5 / 316.3 / 633.3 ms at W=2048/4096/8192 → p0 ≈ 2300 |

Early read (21:20): chip-µs/token-layer SP=2 vs SP=4 — dense@549k 231 vs 295, sparse cold 34 vs 47,
sparse@549k 45 vs 77. Dense row floor drops from ~2944 to ~2300 rows but n=256 still costs as much as n=2048.

## Tools added for Parts B/C (no device)
- `batch_sp2_b.sh` (8 runner sessions), `e2e_bindings/*.yaml`, `e2e_analyze.py` (N ranks, per-boundary hop).
- `m3_budget_sim.py`: default-off `--pipelines`, `--align-recompute`, `--embed-stage0-only`, `--budget-ms`,
  `--latency`, `--split-search`, `--policies`, `--hop-ms`. Default output verified byte-identical.
- `analyze.py`, `additivity.py`, `model_check.py`: layer-set / experiment options; SP=4 outputs verified identical.
- `batch_sp2_c.sh` (scenarios L-a..L-d), `plots_sp2.py`.

## Part A done (22:02) — fit in fit_sp2.txt, coeffs_sp2.json
- dense p0 = 2304 (SP=4: 2944); c per chip identical to SP=4; sparse d per chip 2.8× lower; stages predicted ±7%.
- A3 packing additive within ±3.3%; packed forwards slightly faster than plain on SP=2 → seg_a clamped to 0.
- A5 DRAM in use at 549k (1 user): 10.3–12.6 GB / 34 GB per chip — no memory blocker.

## Part B (running)
- Each session's "warm" producer exits 1: with PREFILL_PRODUCER_CHUNKS=1 the producer loads only 1 chunk of
  prompt tokens, and a 2-chunk warm-up reads past it. Harmless: compile() warms every KV bucket at runner start,
  and measured runs are 48 chunks. Not re-run.

## Done (23:50)
- Part B: 8 runner sessions OK (no hang / OOM). Hop SP=2 11.3 ms, SP=4 9.0 ms. e2e/summary.txt, e2e/model_check.txt.
- Part C: first pass kept in sim_layouts/v1_dense_in_stage0/ (split search kept all dense layers in stage 0,
  which starved L-c at 16.5k); rerun with dense-spread candidates + hill-climb (sim_layouts/split_opt/).
- REPORT_SP2.md written. Recommendation: layout (b), 4 × 4-stage SP=2, split 9,17,17,17, W=4096 + ~290 ms budget.
