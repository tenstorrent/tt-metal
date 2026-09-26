# M3 prefill follow-up — SP=2 vs SP=4 layout study

Branch `vmelnykov/m3_prefill_sp2_study` (from `vmelnykov/m3_prefill_budget_study` `ba334d9`), 2026-09-26,
Blackhole galaxy bh-glx-120-b09u02, bf4 experts, untraced. Everything for this study is in this folder.
Sanity anchor (§2): 15 sparse layers (15–29) on (2,4), W = 4096, cold = **238.2 ms** vs ≈ 241 (−1.2%).

Contents: `runs.csv` / `ops.csv` (same schemas as the first study, `mesh=(2,4) sp=2 ep=8` on the SP=2 rows),
`coeffs_sp2.json` + `fit_sp2.txt`, `additivity_sp2.txt`, `model_check_sp2.txt`, `e2e/summary.txt` +
`e2e/model_check.txt` + per-session logs, `sim_layouts/summary.txt` + one file per scenario, `plots/1–4`,
`PROGRESS.md`. No run hung, OOM'd or timed out.

## Recommendation

**Use layout (b): four independent galaxies, each a 4-stage SP=2 pipeline (4 × (2,4)), split 9,17,17,17**
(stage 0 = the 3 dense layers + 6 sparse), **W = 4096 with a per-forward cost budget of ≈ 290 ms per stage**
(from the 1.5 s hot-latency target). Simulated on the agentic mix: **≈ 39k useful tok/s for 4 galaxies,
+58% over the current plan (a), +45% over (d), +14–20% over (c)**, hot-request latency p50 ≈ 1.4 s. It needs
no cross-galaxy hop at all, so it is also the least exposed to the unmeasured inter-galaxy link. Its one
miss is hot **p99 ≈ 1.95 s** (> 1.5 s): only layout (d) meets p99 ≤ 1.5 s (27k tok/s). The p99 is set by the
deepest requests (history up to 1M, ~100 ms per dense layer each), so fixing it is about dense attention
or routing those requests, not about the layout. Main caveat: on one galaxy the SP=2 pipeline ran 12% below
its cost model on cold traffic (4 ranks sharing one host CPU); on hot traffic the model was 3–7% low.

## Q1 — SP=2 per-layer cost model (Part A)

Fit on A1 + A2 + A2w (+ A5 for the stage overhead), non-negative least squares; `fit_sp2.txt`:

| | a | b | c | d | e | p0 | seg_a | R² |
|---|---|---|---|---|---|---|---|---|
| dense SP=2 | 0.288 | 1.02e-3 | 4.43e-8 | 0 | 0 | **2304** | 0 | 0.9999 |
| dense SP=4 | 1.078 | 4.35e-4 | 2.22e-8 | 0 | 0 | 2944 | 0.61 | 0.9999 |
| sparse SP=2 | 1.962 | 2.46e-3 | 0 | 4.58e-6 | 7.79e-4 | 0 | 0 | 0.9985 |
| sparse SP=4 | 2.020 | 1.36e-3 | 0 | 6.39e-6 | 5.06e-4 | 0 | 0.33 | 0.998 |

Stage overhead o = 1.10e-3 ms/token (SP=4: 0.71e-3); hop 11.3 ms (SP=4: 8.45). Worst residuals: dense W=4096
cold +10%, sparse W=2048 h=16k +6%. The four full SP=2 stages timed alone (A5, 12 points) are predicted within
**±7%**.

Measured chip-µs per token-layer (chips × ms × 1000 / (2048 × layers), W = 2048, n = 2048):

| layer | layout | h = 0 | 141k | 549k |
|---|---|---|---|---|
| dense | SP=4 (4,4) | 21.4 | 96.1 | 295.5 |
| dense | SP=2 (2,4) | 11.8 | 68.4 | 231.1 |
| sparse | SP=4 | 46.9 | 53.2 | 76.3 |
| sparse | SP=2 | 34.5 | 37.5 | 45.5 |

SP=2 is 1.3–1.8× more chip-efficient everywhere. Per chip, dense attention compute is the *same* on both
(c × chips = 3.54e-7 vs 3.55e-7): the dense gain is only the smaller row floor. The sparse history term per
chip is **2.8× lower** on SP=2 (d × chips = 3.7e-5 vs 1.0e-4), and lower per stage too (4.6e-6 vs 6.4e-6).

A4 zone profile (layers 0 + 3, W = 2048, growth from h = 0 to 549k, device ms per layer):
dense `ring_joint_sdpa` +49.8 (SP=4 +33.1); sparse `ag_kv` +1.68 (SP=4 +2.46), `ag_index_k` +0.79 (+1.24),
`indexer` +0.88 (+0.45 — each chip scores twice as many queries). The per-token gather cost halves, as expected.

**Deep hot segment ratio:** per layer, a 2048-row segment at 549k costs 56.0 ms dense vs 2.5 ms sparse on
SP=2 (**~22×**, SP=4 ~10×), because sparse got much cheaper and dense did not. At stage level (3D+12S vs 15S)
it is 5.3× (SP=4, 3D+5S vs 8S: 4.5×). Where the dense layers sit matters even more on SP=2.

## Q2 — dense row floor on SP=2

Still there, smaller: **p0 = 2304 rows** (SP=4: 2944). A 2048-token segment gives each chip 1024 rows instead
of 512, but the kernel still charges ~2300. n = 256 costs exactly what n = 2048 costs at every depth
(549k: 177.0 vs 177.5 ms for 3 layers). Paged / shorter segments still cannot make dense attention cheaper.

## Q3 — packing on SP=2 (A3)

**Additive**: 10 packed forwards (D2, S8'; W = 4096 / 8192; C1, C4, C7, C8, C9) within **±3.3%** of
cold + Σ ΔT1 (worst C7, where the short segment sits first and MoE routes every row; same as SP=4). The cost
model predicts them within 2.1% mean / 5.0% max (`model_check_sp2.txt`). **seg_a ≈ 0**: packed forwards are
slightly *faster* than plain ones at the same W (sparse 125.9 vs 130.0 ms at W = 4096); the fit's negative
value is clamped to 0.

## Q4 — full model on one galaxy (Part B)

Common prefill runner, 60 layers, 48 one-chunk requests per stream, 2d fabric. Hot streams start at
`PREFILL_PRODUCER_PREFIX_TOKENS` = 139264 / 548864 with a **synthetic prefix** (that history KV is never
written; cost is value-independent). tok/s = tokens / (push wall + final ack drain), open loop / K = stages:

| layout | W | cold | hot 139k | hot 549k |
|---|---|---|---|---|
| **4 × (2,4) SP=2** | 4096 | **15.7k** / 14.1k | **14.9k** / 13.4k | **8.3k** / 7.5k |
| 4 × (2,4) SP=2 | 8192 | 16.9k / 15.3k | 15.8k / 14.2k | 8.7k / 8.0k |
| 2 × (4,4) SP=4 | 4096 | 13.7k / 12.3k | 12.9k / 11.6k | 7.9k / 7.1k |
| 2 × (4,4) SP=4 | 8192 | 15.3k / 13.7k | 14.7k / 13.2k | 9.3k / 8.3k |
| 4 × (2,4), split 12,16,16,16 | 4096 | 14.9k / 13.4k | — | **9.2k** / 8.4k |
| 2 × (4,4), split 24,36 | 4096 | 11.8k / 10.5k | — | 7.9k / 7.1k |

- SP=2 wins cold by 15% and hot-139k by 16% (W = 4096). At 549k both are bound by stage 0's dense layers and
  are even at W = 4096; at W = 8192 SP=4 is 7% ahead.
- **Bottleneck** (sync sessions, W = 4096, per-stage compute ms): SP=2 cold 218/256/**270**/255 (a sparse stage),
  hot-549k **532**/310/310/311 (stage 0); SP=4 cold 292/**329**, hot-549k **568**/480.
- **12,16,16,16** lifts SP=2 hot-549k by 11% for −5% cold. **24,36** does not help SP=4 (hot unchanged, cold
  −14%): moving sparse layers only moves the bottleneck to stage 1.
- **Hop** (next stage idle, n = 24 per boundary): **SP=2 11.3 ms** (10.6–11.9), SP=4 9.0 ms (E7: 8.45).
- **Model sanity** (`e2e/model_check.txt`, W / slowest predicted stage vs open loop): SP=4 within ±5% (24,36 hot
  −8%); SP=2 hot +3…+7%, **cold −12%** — the pipelined SP=2 sparse stages run 255–270 ms vs 228 predicted and
  241–250 alone; likely 4 ranks dispatching from one host plus the 2d fabric.
- K = stages reaches 88–97% of open loop.

## Q5 — 4-galaxy projection (Part C)

Simulator (new default-off flags; default output verified byte-identical): `--pipelines` (least-queued-cost
routing), `--align-recompute` (h mod 2048 recomputed: **15.4% extra work** on this mix — the doc's estimate was
~10%), `--embed-stage0-only`, `--budget-ms`, `--latency` (hot latency = (S+1) × forward + (S−1) × hop),
`--split-search` (even, stage 0 = 3…even-share layers, dense spread over the leading stages), plus a hill-climb
(`sim_split_opt.py`) that found no better split for any layout. Agentic mix, n = 4000, measured hops;
`cost` budget from the 1.5 s target (L-a 160, L-b 293, L-c 78, L-d 497 ms):

| | layout | best split | fcfs tok/s (W 4k / 8k) | cost tok/s | cost fwd p50 / p99 | cost hot p50 / p99 |
|---|---|---|---|---|---|---|
| L-a | 8-stage SP=4, 4 galaxies (plan) | 3,9,8,8,8,8,8,8 | 24.7k / 25.1k | 24.6k / 24.9k | 122 / 198 ms | 1.16 / 1.84 s |
| **L-b** | **4 × 4-stage SP=2** | **9,17,17,17** | **39.5k / 41.0k** | **39.0k / 38.5k** | 279 / 383 ms | **1.43 / 1.95 s** |
| L-c | 16-stage SP=2, 4 galaxies | 1,1,1,5,5,5,5,5,4,… | 34.6k / 36.6k | 32.5k | 55 / 102 ms | 1.10 / 1.91 s |
| L-d | 4 × 2-stage SP=4 | 27,33 | 27.0k / 28.7k | 26.9k / 27.1k | 425 / 496 ms | 1.28 / **1.50 s** ✓ |

- **L-b wins throughput by a wide margin**: +58% over the plan. Four independent pipelines keep every stage at
  94–98%; the plan's 8-stage pipeline is capped by its dense stages.
- **Hop 15 ms** (second pass, cross-galaxy stand-in): fcfs tok/s changes by < 0.2% everywhere; under `cost`
  only L-c loses (−2.2%, its budget shrinks from 78 to 75 ms); hot latency grows by ≤ 55 ms (L-c, 15 hops).
  Throughput is not hop-bound.
- L-c needs the dense layers spread one per stage (1,1,1,…); with all three in stage 0 it drops to 16.5k.
- Hot p99 misses 1.5 s for every layout except L-d. The forward p99 is a single very deep request (history
  tail to 1M: ~100 ms per dense layer on SP=2), which no per-forward budget can split; long pipelines multiply
  it by (S+1). The latency estimate is a pessimistic bound (it charges the slowest forward to every stage).
- W: 8192 adds 2–6% under fcfs but doubles forward p50; under the cost budget the two are within 1–2%. Use 4096.

## Q6 — blockers

- **Memory: none.** DRAM in use after the 549k point, 1 user, full stages (A5): SP=2 stage 0 10.3 GB, stages 1–3
  11.4–11.9 GB; SP=4 stages 11.8 / 12.6 GB — of 34.2 GB per chip (weights + KV + persistent gather buffers; not a
  transient peak). The Part B runners (2 users, capacity 557k) ran all 549k streams without OOM.
- **Uneven splits: supported** with no code change (`PREFILL_PP_LAYER_COUNTS`, must sum to the layer count; M3
  imposes no boundary constraint). 12,16,16,16 and 24,36 both ran.
- **Hangs: none** in Parts A–B.
- The producer's 2-chunk warm-up fails with `PREFILL_PRODUCER_CHUNKS=1` (it loads prompt tokens for one chunk
  only); harmless here, since `compile()` warms every KV bucket.
- S8' (layers 8–15) straddles the 4-stage carve's stage 0 (0–14): `BUDGET_ANY_LAYERS=1` allows it.

## What this assumes

- The agentic traffic mix of the first study (independent new / history quantiles, 5% cold, decode takes
  ≤ 1k-token requests at ≤ 64k history). Real prefill histograms would change the absolute numbers.
- Additive per-segment costs (validated on both layouts) and the in-order flowshop pipeline with unlimited
  forwards in flight (E7: K = stages reaches ~90% of that).
- Hops: measured intra-galaxy; cross-galaxy only as the 15 ms sensitivity pass.
- Synthetic hot prefixes in Part B; real-token history in Part A.

## What to measure next

1. The SP=2 cold gap in the pipeline (−12% vs model): host dispatch with 4 ranks per host; a trace path would
   remove it.
2. A split-K dense attention for few-query / long-history segments: it drives stage-0 load, the hot p99 and the
   row floor on both layouts.
3. Selective K/V gathers in MSA (only the top-k blocks): the largest sparse depth term.
4. A cross-galaxy hop measurement, and a routing rule for ultra-deep requests (> ~600k history).
