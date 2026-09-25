# M3 prefill budget study — results (Phase A)

Branch `vmelnykov/m3_prefill_budget_study` from `main` `f5093e7` (includes #57199). 2026-09-25, Blackhole
galaxy bh-glx-120-b09u02, one (4,4) sub-mesh (stage 0 of 2), SP=4 TP=4 EP=16, bf4 experts, untraced.
History: **real tokens** (tiled `longbook_qa_eng_prefill_56320_nopad`), filled chunk by chunk — no synthetic
history in any timed run, so the §E2 synthetic-vs-real validation is not needed. The E3 depth profiles attend
a zeroed cache; their totals match the real-history wall times within the host gap (below).

**Phase B (packed forwards) is not done.** The experiment branch was lost; `main` has no packed path. Q3, the
packed half of Q4, E4, E5-packed and E8 are pending. The simulator numbers below *assume* additivity.

133 rows in `runs.csv` (128 OK, 5 ERROR — see the end), 248 zone rows in `ops.csv`, plots in `plots/`.

## Answers

### Q1 — cold cost vs width (E1, plain B=1)

| layer set | floor (ms, W→0) | slope (ms / 1k tok) | knee |
|---|---|---|---|
| D (3 dense) | 2.9 | 2.06 | none up to 10240 (max resid 13%, W=2048 sits above the line) |
| S8 (8 sparse) | 15.0 | 16.0 | none up to 10240 (max resid 1.4%) |
| S0 (3D + 5S) | 12.3 | 11.1 | none up to 10240 (max resid 7.8%) |

Per layer (fit): dense 1.08 ms + 0.43 µs/tok, sparse 2.02 ms + 1.36 µs/tok (+0.51 µs per *real* token in MoE),
plus a stage overhead of 0.71 µs/tok (embedding; every stage embeds in these runs). §4.4 sanity: S8 at
W = 2048/5120/8192 = 47.8/95.8/146.2 ms vs ~47/92/140 expected. **W = 8192 runs with no workaround:
#57199 fixed the SP=4 hang.** Host gap (1 − device/wall) ≈ 9% cold, ≈ 6% at 549k (2-layer profiles).

### Q2 — depth scaling (E2, E2w, E3)

- **Dense: all depth cost is `ring_joint_sdpa` compute** (0.22 → 8.7 → 33.3 ms per layer at h = 0/141k/549k,
  every other zone flat). Linear in h, and in segment rows with a **floor of ~2944 rows**: a 2048-row segment
  (512 rows/chip) costs as much as ~2944 rows, and n = 256 costs exactly as much as n = 2048. History-only
  share at 549k: **0%**. ~61 TFLOP/s per chip at W = 2048 — the kernel is short of work units, not bandwidth.
- **Sparse: all depth cost is history-only** — `ag_kv` (gather full K+V over SP, 2.55 ms at 549k),
  `ag_index_k` (1.27 ms), `indexer` (0.49 ms); `sparse_sdpa` is flat (and cheaper on the cache-read path than
  at h = 0). Same increment at W = 2048/4096/8192. History-only share at 549k: **100%** at n = 256 and 2048.
- Top 3 zones growing with h — dense: `ring_joint_sdpa` (+33.1 ms), `ccl_out_reduce_scatter` (+0.06),
  `post_attn_norm_allgather` (+0.02). Sparse: `ag_kv` (+2.46), `ag_index_k` (+1.24), `indexer` (+0.45).
- **KV capacity does not matter** (E2c, added): at h = 16k, capacity 18k → 1M changes D by +2%, S8 by +2.5%,
  although the dense ring-gather buffer is sized to capacity.
- The only n dependence is MoE: pad rows are skipped, 0.9–1.3 ms per sparse layer between n = 2048 and 256.

### Q3 — additivity / mixed depths

**Pending Phase B** (needs the packed path). Evidence so far: per-layer costs compose — S0 is predicted from
the D and S8 fits within −7…+10% at every W and h, and three different 5-sparse-layer sets (3–7, 8–12,
11–15) agree within 1–2%. Loop-order and mixed-depth coupling cannot be measured without packing.

### Q4 — forward budget and grouping (simulator, additivity assumed)

- **Budget in ms, not tokens.** At 549k a 2048-row segment adds ~125 ms to stage 0 vs ~28 ms to an
  8-sparse stage; a token budget lets one deep segment stall the pipeline (fwd p99 ~940 ms at W = 8192).
  The `cost` policy (cap each forward at a full cold forward's cost) holds p99 at ~200–260 ms for −1…5% tok/s.
- **Grouping: fcfs ≈ bucket** (within 0.1%); with an additive model grouping changes variance, not work.
- **W:** today no SP=4 limit remains (#57199). tok/s gains are small past 4096 (+2–4% to 8192), p50 latency
  doubles. Recommend W = 4096–8192 with the `cost` cap.

| W (split 8,8,8,8,7,7,7,7) | fcfs tok/s | cost tok/s | cost fwd p99 ms |
|---|---|---|---|
| 2048 | 17.4k | 17.4k | 253 |
| 4096 | 18.6k | 17.5k | 254 |
| 8192 | 19.3k | 18.4k | 257 |

### Q5 — stage 0 and the layer split

Stage-0 ratio S0/S8 (measured / model): 0.79/0.76 at h = 0, 1.22/1.24 at 141k, **2.01/2.12 at 549k**. For
the marginal cost of one deep hot segment the ratio is **4.5×** (dense layer 35.9 ms vs sparse 3.5 ms at
549k per segment) — the prior "~5×" is confirmed for deep hot segments; per new-token segment one dense layer
costs ~10× a sparse layer at 549k (prior estimate 13×). With the default split stage 0 is the bottleneck at 100% while the others idle at
49–59%.

| split (W = 8192, fcfs) | tok/s | stage utilisation |
|---|---|---|
| 8,8,8,8,7,7,7,7 (default) | 19.3k | 100 / 49–55 |
| 6,8,8,8,8,8,7,7 | 22.3k | 100 / 56–64 |
| 4,8,8,8,8,8,8,8 | 26.4k | 100 / 76 |
| **3,9,8,8,8,8,8,8** | **28.7k (+48%)** | 98 / 82–92 |

**Recommend 3,9,8,8,8,8,8,8** (stage 0 = dense layers only). Check that 9 layers + KV fit a (4,4) stage.

### Q6 — padding cost

Pad rows are **not free in attention**: dense attention charges the full padded segment (and at least ~2944
rows); sparse attention cost does not depend on rows at all. Pad rows **are** skipped by MoE (≈0.5 µs per pad
token-layer saved). On the AgentX mix, 2048 rounding gives 82.5% fill; `--paged` (128) raises fill to 98% but
tok/s only +2–4%, because the dense row floor means shorter segments do not get cheaper. Packed pad-row
cost (`0:2048,0:256` vs `0:2048,0:2048`) is pending Phase B.

### Q7 — not run

E6 (SP=2 vs SP=4 at depth) and E7 (2-stage pipeline, hop) not run in Phase A. The simulator's flowshop has no
limit on forwards in flight, so hop_ms = 5/10 changes latency but not tok/s (verified); a bounded in-flight
count is needed before hop can matter.

## Cost model (`coeffs.json`)

```
layer_ms = a + b·W + c·max(p, p0)·h + d·h + e·n        stage_ms = o·W + Σ layer_ms
dense : a 1.078  b 4.35e-4  c 2.22e-8  d 0        e 0        p0 2944   R² 0.9999 (29 pts)
sparse: a 2.020  b 1.36e-3  c 0        d 6.39e-6  e 5.06e-4  p0 0      R² 0.998  (29 pts)
o = 7.09e-4 ms/token (from 8- vs 5-sparse-layer sets; 2·T1 − T2 gives 0–2.5 ms, noisy)
```

p = padded rows of the segment. Fitted by non-negative least squares on E1 + E2 + E2c (smallest capacity) +
E2w. Worst residuals: dense W = 4096 cold +17%, dense W = 2048 cold +12…13% (h = 0 uses the no-cache
attention path, which the model does not separate); sparse W = 2048 h = 16k +8%, cold n = 256 +8%. All
others within ±6%. The simulator now uses `max(padded(n), p0)` for the c term and `o` per token.

## Equivalent-token table (W = 2048, n = 2048)

| h | dense f(h) | sparse f(h) | 60-layer f(h) |
|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 |
| 16k | 1.26 | 0.97 | 0.97 |
| 65k | 2.40 | 1.05 | 1.08 |
| 141k | 4.49 | 1.13 | 1.21 |
| 309k | 8.34 | 1.35 | 1.51 |
| 549k | 13.81 | 1.63 | 1.91 |

## What surprised us / what to measure next

1. **Dense SDPA is work-starved for hot segments.** 512 q-rows per chip do not fill the cores; n = 256 costs
   like n = 2048 and like ~2944 rows. A split-K (flash-decoding) variant of `ring_joint` for few-query /
   long-history segments would cut stage 0's deep-segment cost several-fold. This is the largest single lever
   after the layer split.
2. **~70% of sparse depth cost is gathering the whole K/V history to every chip** (`ag_kv`), though sparse
   SDPA reads 16 blocks per query. Gathering only index_k and then the selected K/V blocks would remove most
   of it. Gathers run at ~58 GB/s.
3. Capacity does not cost time — size caches for the max context.
4. The first layer of a process costs more than later ones at W = 8192 (20.8 vs ~17.3 ms sparse): dispatch
   start-up, visible only in 1–2-layer runs.
5. Next: Phase B packed path → E4/E5/E8; then E6/E7; traced repeats.

## Runs that failed

- `e1_s8_w4096`, `e1_s8_w6144` — ERROR: another session overwrote the harness mid-batch; rerun as `_r2` (OK).
- `20260925_173428_sanity_S8_W2048_p0..p2` — rows written by that other session's harness; not part of the study.
- E3 with `compile()` + real prefix under tracy: the 141k capture wrote a 30 GB ops log and used 198 GB RAM in
  post-processing; stopped before 549k. Replaced by `PROFILE_SKIP_COMPILE=1 SKIP_PREFIX=1` (E3b, ~2 GB each).
- E3 first attempt: `run_prefill_profile.sh` needs `FABRIC` set; no data lost.
- No HANG, OOM or LOAD_TIMEOUT.
