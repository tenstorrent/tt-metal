# M3 prefill budget study — progress

Branch `vmelnykov/m3_prefill_budget_study` from `main` (`f5093e7`, includes #57199). Host bh-glx-120-b09u02.

## §4.0 code recovery
- `vmelnykov/batched_prefill_experiment` is not local, not on origin, not in `/data/vmelnykov/tt-metal`
  (stale July `main`), no Claude transcripts on this host mention it. Owner confirms it is lost.
  → Phase A on `main`; Phase B (packed path) to be rebuilt.

## Setup findings
- Weights / caches on weka: `[4,4]` cache has all 60 layers (~4.1 GB/layer).
- #57199 is in the tree, so no `M3_REPLICATED_TOKENS` (it does not exist on main anyway).
- **Constraint on main:** `actual_start % chunk_size == 0` — history must be a multiple of W, not of 2048.
  The H grid works as-is only at W = 2048; wide forwards at depth need h rounded to a multiple of W.
- **Dense cache-read sizes its ring-gather buffer to the full KV capacity** (`cache_global=max_seq_len`,
  tt/attention/prefill.py). If the op moves/computes over the whole capacity, dense cost follows capacity,
  not h. MSA gathers only the written prefix. → added a capacity sweep (E2c) to Phase A.
- `compile()` warms every chunk bucket up to capacity; the timing harness skips it and warms each point
  with 2 forwards instead (cache-read lengths are runtime args, so no per-h JIT is expected — check in data).

## Tools written
- `models/demos/minimax_m3/tests/perf/budget_sweep.py`: timing harness (no profiler). Progressive real-token
  history fill, `RESULT {json}` lines, knobs `BUDGET_*`.
- `m3_budget_study/run_budget.sh`: reset + load/stall watchdog + log/env capture → `budget_collect.py` → runs.csv.

## §4.4 sanity — PASS
S8 (layers 8–15), (4,4) stage-0 sub-mesh, cold, plain B=1, real tokens: W=2048 / 5120 / 8192 →
**47.8 / 95.8 / 146.2 ms** vs expected ~47 / ~92 / ~140 (+2…4%). W=8192 ran with no workaround:
**#57199 fixes the SP=4 wide-forward hang.** Weight load 40–45 s for 8 layers from weka.
First forward of a new shape pays 3–39 s JIT (incl. a new n — MoE padding config), so 2 warm-ups/point are needed.

## Runs
| run_id | what | status | note |
|---|---|---|---|
| s44_s8_w2048/5120/8192 | §4.4 sanity | OK | 47.8 / 95.8 / 146.2 ms |
| e1_s8_w4096, e1_s8_w6144 | E1 | ERROR | a concurrent session overwrote budget_sweep.py 17:33–17:36; rerun as `_r2` |
| 20260925_173428_sanity_S8_W2048_p0..p2 | — | ERROR | **collision**: rows written by another session's harness, not part of this study; ignore |
| e1_d_w2048 | E1 | OK | n=256 median polluted by warm-up (12.8 → 8.2 ms over the first 5 forwards); first point now gets 5 warm-ups. D shows no n dependence at h=0 (dense MLP ignores actual_isl) |
| e1_* (all) | E1 | OK | cold, plain B=1. S8 linear to 10240 (~16 ms/1k tok), no knee; D ~3× cheaper/token |
| e2c_* | E2c | OK | **capacity does not matter**: D at h=16k 10.28→10.50 ms from cap 18k→1M (+2%), S8 +2.5% |
| e2_d_grid, e2_s8_grid, e2_s0_check | E2 | OK | real-token history up to 549k; D no n dependence; S0/S8 = 0.79 / 1.22 / 2.01 at h = 0 / 141k / 549k |
| e2w_* | E2w (added) | OK | same depths at W=4096/8192: dense ∝ max(rows, ~2900)·h, sparse depth term ~history-only |

## Preliminary model (analyze.py, E1+E2+E2c+E2w, non-negative LS)
- dense  layer_ms = 1.08 + 6.7e-4·W + 2.22e-8·max(p, 2944)·h          R² 0.9999
- sparse layer_ms = 2.02 + 1.45e-3·W + 6.4e-6·h + 5.1e-4·n              R² 0.998
- Worst residuals ≤12%, all at h=0 (no-cache attention path differs from cache-read path).
- Dense attention has a row floor: 2048-row segments (512 rows/chip) cost like ~2944 rows — cores
  under-filled. n=256 costs the same as n=2048. Paged/smaller segments cannot help dense without a
  split-K kernel.
- "Stage overhead" o = T_S8 − 8/5(T_S0 − T_D) grows with W and h (0.4 → 14 ms): not an overhead but
  layer heterogeneity (layers 8–15 heavier than 3–7?). Checking with batch_layers.sh (lh_*).

## Preliminary simulator (assumes additivity; Phase B pending)
- Stage 0 is the bottleneck at 100% for the default split; others 48–62%.
- Split 3,9,8,8,8,8,8,8 (dense-only stage 0): 28.7k vs 19.1k tok/s (+50%) at W=8192 fcfs.
- fcfs ≈ bucket; cost policy cuts fwd p99 ~940 → ~200–260 ms for −1…5% tok/s. W>4096 adds 2–4%.
- --paged: +2–4% (fill 82 → 98%). dec-max-hist 16k vs 65k: no change.
| lh_* | per-layer check (added) | OK | sparse layers 3–7 / 8–12 / 11–15 equal within 1–2% |
| e3_*_r2 | E3 (compile+prefix under tracy) | 2 OK, stopped | 141k capture: 30 GB ops log, 198 GB RAM in post-processing; stopped |
| e3b_* | E3 (PROFILE_SKIP_COMPILE + SKIP_PREFIX) | OK | 6 profiles, ~2 GB each; op breakdown in REPORT Q2 |
| o_* | stage overhead (added) | OK | 2·T1−T2 noisy 0–2.5 ms; o = 0.71 µs/tok from 8 vs 5 layers |

## Status (20:15)
Phase A done: coeffs.json, sim/, plots 1/2/2b/3/5, REPORT.md. Phase B (packed path; E4, E5-packed, E8,
Q3) not started — needs model-code work, waiting on the owner's go-ahead.
