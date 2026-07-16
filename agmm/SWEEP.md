# AGMM / MM block-size sweep + perf-tracking harness

A shape-list-driven harness for tracking the best achievable perf of the fused
**AllGather + Matmul** op (and plain `minimal_matmul`) across a curated list of
shapes, over time and across commits. You maintain one editable list of shapes;
the runner sweeps each on device, records the best blocking and its perf, and
appends to an append-only history so you can always see *the current best on
latest main* and how it moves as optimizations land.

## Files

| File | Role |
|---|---|
| `sweep_shapes.json` | **You edit this.** The list of shapes/configs you care about. |
| `run_sweeps.py` | The runner. Consumes the spec, sweeps each shape, writes history. |
| `roofline_lib.py` | Shared roofline math (`compute_roofline`); used by the runner and `roofline.py`. |
| `build_dashboard.py` | Regenerates the `agmm_db.html` dashboard from `sweep_latest.csv` + `roofline_lib`. Pure Python (no ttnn); run_sweeps calls it automatically. |
| `sweep_history.csv` | **Output.** Append-only, one row per (shape, run). The tracking record. |
| `sweep_latest.csv` | **Output.** Regenerated each run: most-recent best per shape (at-a-glance). |
| `agmm_db.html` | **Output.** Self-contained dashboard: sortable/filterable roofline table + per-shape drawer. Refreshed each run; open in a browser. |
| `sweeps/<commit>_<ts>/` | **Output.** Full per-combo CSV per shape (drill-down; git-ignored). |
| `gen_isolated_specs.py` | Regenerates `isolated_{ag,mm}_spec.json` from `sweep_shapes.json`. Run after editing the shape spec. |
| `isolated_{ag,mm}_spec.json` | **Generated.** Per-op specs: AG (`op_type=ag`, deduped by device/M/K) and MM (`op_type=mm`, one per shape). |
| `run_isolated.sh` | Driver: runs the AG sweep then the MM sweep into their own `isolated_*` CSVs. Path-relative (works from any checkout). |
| `build_comparison.py` | Joins fused AGMM vs serial AG+MM → `comparison.csv` (fusion savings + overlap ratio). |
| `isolated_{ag,mm}_{history,latest}.csv`, `comparison.csv` | **Output.** Isolated-sweep tracking records + the fused-vs-serial join. |

The runner reuses the on-device sweep machinery in
`models/tt_dit/utils/sweep_mm_block_sizes.py` — it does **not** reimplement the
block search, profiler capture, or ops-log parsing. The isolated sweeps use the
same machinery via `op_type` (`ag` = `all_gather_async`, `mm` = `minimal_matmul`).

## Quick start

Run from the repo root inside the tt-metal python env (needs `ttnn` + `tracy`).
On the Blackhole galaxy this goes through the tt-device-mcp broker; the raw
command is:

```bash
TT_METAL_HOME=$PWD PYTHONPATH=$PWD python_env/bin/python agmm/run_sweeps.py [flags]
```

```bash
# 1. (once) establish the best blocking per shape — exhaustive, slow (~minutes/shape)
python agmm/run_sweeps.py --mode full

# 2. (routine) track perf on latest main — measures only the known-best blocking, fast
python agmm/run_sweeps.py            # --mode track is the default

# subset by id or tag
python agmm/run_sweeps.py --ids s2_compute_n1024 s2_ff_chunks3_n3072 --mode full
python agmm/run_sweeps.py --tags stage2

# regenerate the whole 20-shape spec from an agmm_instances.json extraction
python agmm/run_sweeps.py --seed-from-instances
```

> **Device-time note.** A `full` sweep JIT-compiles every candidate blocking
> (~2s each; 150–270 candidates/shape ⇒ ~5–8 min/shape; the full 7-shape list is
> **>1 hr** of device time). On a shared device, prefer one shape at a time
> (`--mode full --ids <one>`). `track` runs are seconds/shape.

## Isolated AG / MM sweeps (fused-vs-serial)

Break each fused AGMM shape into its two halves to see what fusion buys. Run on a
**healthy 2-link galaxy** — a node with a degraded fabric link deadlocks the
gather (see `GALAXY_FABRIC_FINDINGS.md`).

```bash
# 1. (re)generate the per-op specs from sweep_shapes.json (only after editing shapes)
python agmm/gen_isolated_specs.py

# 2. run both isolated sweeps -> isolated_{ag,mm}_{history,latest}.csv
bash agmm/run_isolated.sh            # AG (op_type=ag) then MM (op_type=mm), ~2 h full

# 3. join fused vs serial -> comparison.csv (+ printed table)
python agmm/build_comparison.py
```

`build_dashboard.py` folds the isolated AG/MM times into every card automatically
on its next run. The isolated all-gather is deduped by `(device_config, M, K)`
(independent of N/fusion); the matmul keeps N + fusion so it matches AGMM's
fused matmul. `overlap = AGMM ÷ (AG + MM)`: `<1` means fusion wins, `>1` means the
fused op's fixed overhead beats the benefit (seen on small-M shapes).

## How it works

### 1. The shape spec (`sweep_shapes.json`)

A flat JSON array of shape records. One record:

```json
{
  "id": "s2_compute_n1024",          // stable unique key — history joins on this
  "op_type": "agmm",                 // "agmm" (all_gather_minimal_matmul_async) | "mm" (minimal_matmul)
  "device_config": "bh_4x8",         // key into DEVICE_CONFIGS in the sweep script
  "M": 4864, "K": 4096, "N": 1024,   // K is the GATHERED K for agmm (= K_local * ring_size)
  "grid": [12, 9],                   // compute grid [x, y]
  "dtype": "bfloat16",
  "math_fidelity": "HiFi2",
  "fusion": {                        // all optional; omit or {} for a plain matmul
    "chunks": 3,                     //   split the op into N chunks
    "math_approx_mode": true,        //   approximate math LLK
    "fused_activation": "gelu",      //   "gelu" | "gelu_approx" | "silu" | "relu"
    "use_addcmul": true,             //   fuse an addcmul on the output
    "scalar": 1.0,                   //   addcmul scalar
    "use_matmul_split": false        //   (mm only) use minimal_matmul_split
  },
  "tags": ["stage2", "compute-limited"],   // free-form; filter with --tags
  "notes": "..."
}
```

Rules of thumb:
- **`id` is permanent** — history rows key on it. Never reuse an id for a
  different shape, or you'll splice unrelated data points into one trend.
- **`K` is the gathered K** for `agmm` (matches the `K_gat` column in
  `AGMM_roofline_analysis.md`); the runner derives per-device `K_local` itself.
- The `fusion` knobs are plain strings/bools so the spec stays tool-agnostic;
  the sweep worker maps them to `ttnn` enums.

To add a shape: append a record. To seed the full set of 20 shapes extracted
from the two 32-device dumps, run `--seed-from-instances` (writes over
`sweep_shapes.json`).

### 2. The two modes

Warmup dominates cost: the sweep compiles a kernel per candidate blocking. So
the runner separates *searching* for the best blocking from *tracking* it.

- **`--mode full`** — generates the full block-size candidate grid (even sizes +
  divisors for M/N blocks, divisors of `K_local` for K block, `2×2` subblock
  under fp32-dest), pre-filters by an L1 budget, compiles + trace-measures every
  survivor, and records the **min-duration** blocking. Run occasionally.
- **`--mode track`** (default) — looks up each shape's most-recent `OK` best
  blocking in `sweep_history.csv` and measures **only that one blocking**
  (via `MM_SWEEP_EXPLICIT_COMBOS`). Seconds per shape. A shape with no prior
  best falls back to a full sweep automatically. Run per-commit to watch main.

### 3. Per-shape execution

For each selected shape the runner:
1. Serializes the record into `MM_SWEEP_SHAPE_JSON` and invokes
   `test_mm_sweep_worker_external` under the device profiler
   (`tracy.process_model_log.run_device_profiler`), with mid-run profiler dumps
   so the buffer doesn't overflow across many combos.
2. The worker opens the mesh once, sweeps the candidate blockings (or the single
   tracked one), and writes the combos that survived warmup to a temp file.
3. The runner parses the profiler ops-log, aligns durations to combos, and picks
   the minimum as the best blocking.
4. For `agmm` shapes it computes the roofline (`compute_roofline`) and joins
   `ideal_us` / `limiter` / `speedup` onto the result.
5. Appends one row to `sweep_history.csv` (written incrementally, so a crash or
   kill only loses the in-flight shape), and archives the full per-combo CSV.

After all shapes, `sweep_latest.csv` is regenerated (most-recent best per shape).

### 4. The roofline join

`roofline_lib.compute_roofline(M, K_gathered, N, ring_size, num_links, grid,
math_fidelity, time_us)` returns the best-case `ideal_us`, the binding resource
`limiter` (compute / dram / fabric), and — given a measured time — the
achieved utilizations and `speedup = measured / ideal`. The efficiency ceilings
are now **100% of peak** on every resource (FLOP, DRAM BW, fabric BW) — `ideal`
is the hard physical roofline, so `speedup` is the full gap to peak hardware.
Both this harness and `roofline.py` import it so there is one source of truth.
Each history row is therefore self-contained:
measured best **and** distance-to-ideal for that commit.

## Output schema (`sweep_history.csv`)

One row per (shape, run), append-only. Columns:

```
timestamp, git_commit, git_branch, dirty, mode,
shape_id, device_config, op_type, M, K, N, grid, fusion_summary,
best_M_block, best_K_block, best_N_block, best_sb_h, best_sb_w,
best_duration_ns, best_duration_us,
ideal_us, limiter, speedup,          # agmm only
n_measured, n_skipped, status, sweep_csv_path
```

- `dirty` flags an uncommitted working tree (WIP sweeps are allowed but marked).
- `status` is `OK`, or an error tag (`SWEEP_FAILED`, `NO_COMBOS`, `NO_TIMINGS`)
  for rows where the run didn't produce a timing.
- Filter by `shape_id` and plot `best_duration_us` against `timestamp` /
  `git_commit` to see whether an optimization moved a given shape.

`sweep_history.csv` and `sweep_latest.csv` are **not** git-ignored — commit them
if you want the perf record versioned in the repo. The bulky `sweeps/` archive
is git-ignored (local drill-down only).

## Prerequisites

The device-profiler path (`python -m tracy`) requires the `websockets` import to
be lazy in `tools/tracy/serve_wasm.py` (fixed on this branch); otherwise the
profiler subprocess dies at import before any timing is captured.
