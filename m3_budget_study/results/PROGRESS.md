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
