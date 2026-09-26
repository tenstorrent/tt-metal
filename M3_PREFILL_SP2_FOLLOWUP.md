# MiniMax-M3 prefill follow-up: SP=2 vs SP=4 layout study on one galaxy

**Audience:** the same kind of engineering agent that ran `M3_PREFILL_BUDGET_STUDY.md`, on one Blackhole galaxy.
**Owner:** Slava Melnykov.
**Builds on:** branch `vmelnykov/m3_prefill_budget_study` and its results (`REPORT.md`, `coeffs.json`, the tools listed in §2).

Read this whole file first, then work in the order given. Keep `results_sp2/PROGRESS.md` up to date as you go.

---

## 0. Why this study

The first study fitted a cost model for the planned layout: 8 pipeline stages of **(4,4) sub-meshes, SP=4**, over 4 galaxies. It projected about **24.6k tok/s for 4 galaxies** on the agentic traffic mix. That is about 6.2k per galaxy, against ~18–19k per galaxy measured earlier for cold, full segments on a single galaxy with SP=2.

Part of that gap is the layout. In E6, SP=2 needed **1.2–1.27× less chip time per token cold, and 1.36–1.53× less at 549k history**. The reason: every chip gathers the whole K/V and index_k history in each sparse layer, and SP=4 pays that on twice as many chips. The dense-attention row floor (~2944 rows) may also shrink on SP=2, because a 2048-token segment gives each chip 1024 rows instead of 512.

E6 only covered sparse layers on the plain path. We need a full SP=2 cost model and a full-model head-to-head before choosing the layout.

**Decision this study feeds:** for 4 prefill galaxies, which layout to use:

- (a) one 8-stage SP=4 pipeline (the current plan);
- (b) 4 independent galaxies, each a 4-stage SP=2 pipeline;
- (c) one 16-stage SP=2 pipeline across the 4 galaxies.

## 1. Questions to answer in `results_sp2/REPORT_SP2.md`

| # | Question | Part |
|---|---|---|
| Q1 | Per-layer cost model on (2,4) SP=2 (TP=4, EP=8), for dense and sparse layers: width, history depth, new tokens. Compare with the SP=4 `coeffs.json` in chip-µs per token-layer at h = 0 / 141k / 549k, per layer type. | A |
| Q2 | Does the dense-attention row floor change on SP=2? Compare n = 256 vs 2048 at depth. What is the fitted `p0`? | A |
| Q3 | Is packing still additive on SP=2, and what is the per-segment overhead (`seg_a`)? A spot check is enough. | A |
| Q4 | Full model on one galaxy, with the common prefill runner: 4 × (2,4) vs 2 × (4,4). Cold and hot tok/s, bottleneck stage, and hop time. | B |
| Q5 | Projection for 4 galaxies on the agentic mix: layouts (a), (b) and (c). For each: throughput, forward-time p50/p99, hot-request latency, best layer split, recommended W or ms budget. Which layout wins, and by how much? | C |
| Q6 | Blockers: memory at 549k history, whether the runner supports uneven layer splits, hangs, anything else. | all |

## 2. Rules, tools and setup

The rules are unchanged from the first study:

- Work on a new branch `<you>/m3_prefill_sp2_study` created from `vmelnykov/m3_prefill_budget_study`.
- Add default-off knobs only; no kernel or C++ changes.
- Run `tt-smi -glx_reset` before every process.
- Timeouts: 20 min including load, and 5 min with no progress.
- `runs.csv` is append-only; failed runs get rows too.
- Record env and SHA per run.

Also:

- **Only one session may use the harness.** Last time another session overwrote `budget_sweep.py` mid-batch. Create a lock file (`results_sp2/.lock` with your PID) and check it before every batch.

**Reuse the tools you wrote:**

- `models/demos/minimax_m3/tests/perf/budget_sweep.py` (knobs `BUDGET_*`)
- `m3_budget_study/run_budget.sh`, `budget_collect.py`, `analyze.py`
- the extended simulator (`p0`, `o`, `seg_a`, `hop_ms`)
- the packed path (`segment_size`, `prefill_segments`)
- the producer knobs `PREFILL_PRODUCER_MAX_IN_FLIGHT` and `PREFILL_PRODUCER_PREFIX_TOKENS`

**Mesh and cache:**

- **(2,4) sub-mesh:** carve it with `STAGES=4 STAGE=k`: SP=2, TP=4, EP=8.
- **Tilized cache:** the `[2,4]` cache on weka, under `/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/`.
- **Sanity anchor from E6:** 15 sparse layers (15–29) on (2,4), cold, took ≈ 241 ms at W = 4096, i.e. 31.3 chip-µs per token-layer. Reproduce that first. Stop if you are off by more than 15%.

Keep the same CSV schemas. Set `mesh="(2,4)"`, `sp=2`, `ep=8` on the new rows so they can be merged with the SP=4 data.

---

## 3. Part A: SP=2 per-layer cost model (P0, single stage)

Run everything on one (2,4) sub-mesh with real-token history, filled progressively as before.

**Layer sets:**

- **D2**: layers 0–2 (3 dense).
- **S8'**: layers 8–15 (8 sparse). Same layers as S8, so the two layouts compare layer for layer.

| Sweep | Grid | Purpose |
|---|---|---|
| A1 cold width | D2, S8': W ∈ {2048, 4096, 8192, 10240}, plain B=1, h=0 | floor and slope (a, b) |
| A2 depth | D2, S8': W=2048, n ∈ {256, 2048} × h ∈ {0, 16384, 65536, 141312, 309248, 548864} | c, d, e, and the dense row floor `p0` (Q2) |
| A2w wide depth | D2, S8': W ∈ {4096, 8192}, B=1, h ∈ {0, 139264, 548864} (h must be a multiple of W on the plain path) | row dependence of the dense term |
| A3 packing | S8' and D2 at W=4096: C1 `0:2048,0:2048`, C4 `548864:2048,0:2048`, C7 `548864:1024,0:2048`. At W=8192: C8 (4 × cold), C9 (1 deep + 3 cold) | additivity and `seg_a` (Q3) |
| A4 profile | zone profiler `STAGES=4 STAGE=0 LAYER_IDS=0,3`, W=2048, n=2048, h ∈ {0, 141312, 548864}, with `PROFILE_SKIP_COMPILE=1 SKIP_PREFIX=1` as in E3b | which zones grow with h on SP=2 |

**Fit:**

- Fit `coeffs_sp2.json` in the same form as `coeffs.json`: `a, b, c, d, e, p0, seg_a`, the per-token overhead `o`, and `hop_ms` from Part B.
- Report R² and the worst residuals.
- Tabulate chip-µs per token-layer for SP=2 against SP=4, per layer type, at h = 0 / 141k / 549k.

**What to look for:**

1. **Dense row floor (Q2).** Is n=256 still as expensive as n=2048 at depth on SP=2? Does `p0` drop from ~2944?
2. **Sparse gathers.** Does SP=2's per-chip history gather cost roughly half of SP=4's per token processed?
3. **Deep hot segment ratio.** Does the dense-vs-sparse ratio for a deep hot segment stay around 4.5×?

---

## 4. Part B: full-model head-to-head on one galaxy (P0, common runner)

This is the most direct evidence. Use the existing manifests to run the full 60 layers two ways on the same galaxy:

- **4 × (2,4):** `models/demos/minimax_m3/tt/runners/manifests/m3_binding_mock_migration_intragalaxy_4rank.yaml`. 15 layers per stage, SP=2.
- **2 × (4,4):** `models/demos/minimax_m3/tt/runners/manifests/m3_binding_mock_migration_intragalaxy_2rank.yaml`. 30 layers per stage, SP=4.

Per galaxy, the 2 × (4,4) run is what each galaxy of the planned 8-stage SP=4 layout does. The 4 × (2,4) run is layout (b).

**Streams**, as in E7: 48 chunks per run, each chunk its own request.

- **Width:** W ∈ {4096, 8192}.
- **History:**
  - **cold:** h = 0;
  - **hot-141k:** every chunk at h = 139264;
  - **hot-549k:** every chunk at h = 548864.

  Use `PREFILL_PRODUCER_PREFIX_TOKENS`, and state whether the prefix KV is real or synthetic.
- **In flight:** K = number of stages (4 or 2), and open loop.

**Record, per run:**

- tok/s = tokens / (push wall + final ack drain);
- per-stage compute time (median) and which stage is the bottleneck;
- hop time, measured in a synced session on chunks where the next stage was idle;
- peak DRAM per chip.

Put all of it in `results_sp2/e2e/summary.txt`, plus logs.

**Layer split.** If the runner supports an uneven per-rank layer split (check the manifest schema and the adapter), also run:

- **4 × (2,4):** 12,16,16,16, i.e. a lighter stage 0 that holds the 3 dense layers;
- **2 × (4,4):** 24,36.

Run each on hot-549k and cold at W=4096. If the runner doesn't support it, say so and leave it for the simulator.

**Sanity:** compare each measured tok/s with the per-stage cost model (`coeffs.json` for SP=4, `coeffs_sp2.json` for SP=2) and report the error.

---

## 5. Part C: 4-galaxy layout projection (P1, simulator)

First extend the simulator. These are default-off flags; existing outputs must not change.

1. `--coeffs` per scenario (SP=4 vs SP=2 files).
2. `--pipelines P`: P independent pipelines. The router sends each request to the pipeline with the least queued predicted cost; round-robin is acceptable if documented. Throughput = total useful tokens / max makespan.
3. `--align-recompute`: a request whose history h is not a multiple of 2048 recomputes `h mod 2048` old tokens, and they count as work but not as useful tokens. This is today's behaviour on the packed path. The first study's projections left it out, and it costs about 10%.
4. `--embed-stage0-only`: charge the per-token embedding overhead `o` to stage 0 only, as in production.
5. Report the **hot-request latency estimate**: `(stages + 1) × slowest-forward time + (stages − 1) × hop`, as p50 and p99.
6. Report the **best split per scenario**, searched over a small set that includes "stage 0 = dense layers only". For 16 stages, try stage 0 = 3 layers with the rest ≈ 57/15, plus one or two alternatives.

**Scenarios.** All use `--align-recompute --embed-stage0-only`, the agentic mix as before, and `--n 4000`.

| ID | Layout | Coeffs | Stages × pipelines |
|---|---|---|---|
| L-a | 8-stage SP=4 over 4 galaxies (current plan) | `coeffs.json` | 8 × 1 |
| L-b | 4 independent 4-stage SP=2 galaxies | `coeffs_sp2.json` | 4 × 4 |
| L-c | 16-stage SP=2 over 4 galaxies | `coeffs_sp2.json` | 16 × 1 |
| L-d | 4 independent 2-stage SP=4 galaxies | `coeffs.json` | 2 × 4 |

For each scenario:

- run W ∈ {4096, 8192}, both `fcfs` and `cost` with a stage budget derived from a **1.5 s hot-latency target**, i.e. `budget_ms = (1500 − (stages − 1) × hop) / (stages + 1)`;
- report tok/s, forward-time p50/p99, hot latency p50/p99, stage utilisation and the best split;
- run a second pass with a hop of 15 ms to cover cross-galaxy hops, which are unmeasured.

Add one plot: tok/s against hot-latency p99, one point per scenario and policy.

---

## 6. Order and time

1. **Sanity check (§2)**: E6 anchor on (2,4).
2. **Part A:** A1, then A2, A2w, A3, A4. Fit `coeffs_sp2.json`.
3. **Part B:** 4 × (2,4), then 2 × (4,4); the split variants only if supported.
4. **Part C:** simulator extensions, then scenarios.
5. **Report.**

Stop and write to the human if:

- the sanity anchor is off by more than 15%;
- 3 runs in a row hang;
- a full-model run at 549k runs out of memory;
- an uneven split needs a runner change beyond a default-off knob.

## 7. Deliverables (`results_sp2/`)

- **Data:** `runs.csv`, `ops.csv` (same schemas).
- **Model:** `coeffs_sp2.json`, `fit_sp2.txt`.
- **End-to-end:** `e2e/summary.txt`, plus logs.
- **Simulator:** `sim_layouts/*.txt`, one file per scenario.
- **Plots** (`plots/`):
  1. layer ms vs h, SP=2 vs SP=4, dense and sparse panels;
  2. chip-µs per token-layer vs h, per layer type and layout;
  3. full-model e2e tok/s, cold vs hot, 4 × (2,4) vs 2 × (4,4);
  4. tok/s vs hot latency per 4-galaxy layout.
- **`REPORT_SP2.md`:**
  - Q1–Q6 answered with tables;
  - a one-paragraph recommendation: layout, split, W or ms budget;
  - what it assumes, and what to measure next.
