<div align="center">

<picture>
  <img alt="tt-ember logo" src="docs/images/tt_ember_logo.png" height="220">
</picture>

**Power and energy telemetry for Tenstorrent hardware.**

</div>
<br>

tt-ember collects and analyzes power metrics (voltage, current, power) from Tenstorrent hardware during workload execution, built on [TT-Metalium](https://github.com/tenstorrent/tt-metal). It produces per-interval statistics, charge and energy calculations, and matplotlib visualizations — all aligned to the execution windows of the measured application. An optional device-side layer correlates per-kernel activity (reader / compute / writer) with the same power telemetry for a wait-vs-active-vs-overhead breakdown.

-----
**Contents:** [Overview](#overview) · [Repository Structure](#repository-structure) · [Prerequisites](#prerequisites) · [Quick Start](#quick-start) · [Scripts](#scripts) · [Output](#output-structure) · [Benchmark App](#the-benchmark-application) · [Power Metrics](#power-metrics)

-----

## Overview

tt-ember runs a measured application (by default the `long_matmul` programming example) while a telemetry binary samples the device's power sensors, then joins the two by wall-clock time to attribute power and energy to each phase of execution. One driver, `run_sweep.py`, runs any sweep described by a YAML config under `sweeps/`; `auto.py` is the single-measurement building block it calls. No build step for the tooling itself; only the `tt-metal` binaries it drives need to be compiled.

## Repository Structure

```
tt-ember/
  run_sweep.py         Sweep driver — reset + auto.py per run + analyses, from a YAML config
  sweeps/              One config per result: paper_fig1_stage_breakdown, paper_fig2_blocked,
                       crossop, decoder_block, prefill
  auto.py              Orchestrator — runs telemetry + app + parser in sequence
  parser.py            Analysis engine — metrics, CSV, figures
  compare_runs.py      Cross-run comparison — overlays metrics across runs
  compare_cases.py     Named use-case comparison — grouped bar charts per grid
  common.py            Helpers shared by the scripts (reset, run bookkeeping, measurement mode)
  analysis/
    power_cases.py     Per-engine energy ablation and cross-op charts (run_sweep.py steps)
    make_pj_per_flop.py, make_pj_per_flop_by_engine.py,
    make_pj_per_flop_naive_vs_blocked.py, crossop_corrected_flops.py
  aiclk/               Pin Blackhole's AICLK for iso-clock sweeps (set_aiclk.py)
  op_power_breakdown/  ttnn decoder-block workload and its per-op breakdown
  docs/
    device-side-instrumentation.md   Per-kernel (reader/compute/writer) analysis
```

## Prerequisites

### Python environment

Activate a Python environment that has tt-metal's Python dependencies (`tt_umd`, etc.) available:

```bash
source /path/to/tt-metal-venv/bin/activate
```

### Measurement mode

`LONG_MATMUL_MODE` selects how runs are measured and analysed, for `long_matmul` and for these
scripts alike: `updated` (the default) or `legacy`, which reproduces how the published tt-ember
results were measured. A `run_sweep.py` config sets it with `mode:`, for the runs and the
analyses alike. The other scripts inherit it from the environment, so to recreate those results
by hand run `export LONG_MATMUL_MODE=legacy` first; `parser.py` and the `analysis/` scripts also
take `--mode`. See
`long_matmul`'s README for what each mode changes in the workload; in the analysis, `legacy`
keeps the idle-pause window widened by the trim, floors dynamic current at 0, and uses the
trimmed window for pJ/FLOP.

Measure power and profile kernels in separate runs: with `TT_METAL_DEVICE_PROFILER=1` the
kernels read the timer in their inner loops and the profiler is read back during the first idle
pause, which is the one the base current is measured in. Unset it before a power sweep.

### Built binaries

The following binaries must be compiled from the [`tt-metal`](https://github.com/tenstorrent/tt-metal) repository before running the pipeline. Point `$TT_METAL_HOME` at your checkout:

```bash
export TT_METAL_HOME=/path/to/tt-metal
```

```
$TT_METAL_HOME/build_Release/tools/umd/telemetry
$TT_METAL_HOME/build_Release/programming_examples/metal_example_long_matmul
```

`./build_metal.sh --build-programming-examples` builds both. Run the scripts below from
`$TT_METAL_HOME/tools/tt-ember`.

## Quick Start

Reproduce one result of the Wormhole-vs-Blackhole energy paper with one command. Each config
under `sweeps/` lists the runs (which `POWER_CASE`, op and block size) and the analyses that
turn them into the figure:

```bash
python3 run_sweep.py --config sweeps/paper_fig1_stage_breakdown.yaml \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --app-exe "$TT_METAL_HOME"/build_Release/programming_examples/metal_example_long_matmul \
  --tt-venv-activate /path/to/venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out_fig1
```

| Config | Paper result | Runs |
|---|---|---|
| `paper_fig1_stage_breakdown.yaml` | Table 2, Table 3, Fig. 1 | `POWER_CASE` 0, 1, 2, 4, 5 on matmul |
| `paper_fig2_blocked.yaml` | Fig. 2 | `POWER_CASE` 0, block 1x1 vs 2x4 |
| `crossop.yaml` | cross-op sweep (not in the paper) | cases 5, 1, 2, 4 × seven ops |
| `decoder_block.yaml` | Table 4, Figs. 3–4 | `op_power_breakdown/ttnn_ops_workload.py` |
| `prefill.yaml` | the original `run_all.py` prefill measurement | `POWER_CASE` 0 |

Runs land in `<output-root>/<op>/<case>[_b<M>x<N>]/` (or `<output-root>/<subdir>/` for the
ttnn workload), so every analysis script also works on output produced by hand with `auto.py`.

### One measurement by hand

Run one full measurement — telemetry capture, application, and analysis — in a single command:

```bash
python3 auto.py \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --telemetry-freq 50 \
  --app-exe "$TT_METAL_HOME"/build_Release/programming_examples/metal_example_long_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out \
  --subdir prefill_1024_2048_2048_160 \
  --slot-ms 1 \
  --device-id 0 \
  --trim-ms 1.0 \
  --app-args 1024 2048 2048 160
```

> `--app-args` must always be placed **last** — it uses `argparse.REMAINDER` and will consume every argument that follows it.

Results land in `./out/prefill_1024_2048_2048_160/` (see [Output Structure](#output-structure)).

## Scripts

### `auto.py` — Full Pipeline Orchestrator

Manages the full measurement lifecycle: starts the telemetry binary as a background subprocess, runs the application in the foreground while capturing its stdout to `summary.txt`, waits for a 10-second cool-down, and invokes `parser.py` on the captured data. All output is written to a structured directory under `--output-root`.

It returns the app's or the parser's exit code when one of them fails, and its own codes otherwise: 20 (telemetry exited right after starting), 21 (no telemetry file), 22 (the app wrote no output), 23 (telemetry exited before the run ended). The sweeps print what each of these and the parser's 10 and 11 mean.

### `parser.py` — Analysis Engine

Can be run standalone against a previously captured telemetry log. Parses raw voltage, current, and power samples, bins them into time slots, computes per-interval metrics, writes a 29-column CSV, and produces 13 matplotlib figures (with `--program-log`; without it, only `telemetry_overview.png`). It exits with code 10 when the outputs are written but a run is unusable (a grid with no telemetry samples, or no idle pause to measure the base current in), and with code 11 for telemetry from several devices unless `--device-id` says which one to use.

```bash
python3 parser.py \
  -i ./out/prefill_1024_2048_2048_160/telemetry.txt \
  --program-log ./out/prefill_1024_2048_2048_160/summary.txt \
  -o ./out \
  --subdir prefill_1024_2048_2048_160 \
  --slot-ms 1 \
  --device-id 0 \
  --trim-ms 1.0
```

### `run_sweep.py` — Sweep Driver

Runs every measurement of a sweep by invoking `auto.py` once per run, then the analyses. Before
each run it issues a hardware reset via `tt-smi -r` and waits for the device to re-initialize,
ensuring a clean thermal and electrical baseline; with `aiclk_mhz` set in the config it re-pins
the Blackhole clock after every reset (see `aiclk/`). Stops immediately if any reset or run fails.

- A run is skipped once it is complete: `auto.py` writes `run_complete.json` into the run
  directory only after the parser succeeded, recording the app arguments, the workload
  environment variables (including `LONG_MATMUL_MODE`, `TT_METAL_DEVICE_PROFILER`,
  `TT_METAL_WATCHER` and `TT_METAL_DPRINT_CORES`) and the telemetry settings. So an interrupted
  sweep resumes, and a run made with different settings stops the sweep instead of being mixed
  in; use another `--output-root`, or `--force` to re-run.
- The analyses read only runs complete with this sweep's settings.
- `--dry-run` prints every command without touching the board; `--analyses-only` regenerates the
  charts from existing runs.
- Every command and its environment goes to `<output-root>/sweep.log`; the config, the tt-metal
  commit and the board list go to `<output-root>/PROVENANCE.md`.

A config is a YAML mapping:

```yaml
name: paper_fig1_stage_breakdown
workload: long_matmul        # or ttnn_ops
mode: legacy                 # LONG_MATMUL_MODE for the runs and the analyses; default updated
app_args: [1024, 2048, 2048, 160]  # M N K iters [fixed_blocks_per_core]
telemetry_freq_hz: 20000           # a 50 us polling period, as the published captures used
trim_ms: 1.0
aiclk_mhz: null                    # e.g. 1000 for an iso-clock Blackhole sweep
runs:
  power_cases: [0, 1, 2, 4, 5]     # one run per (op, case, block) combination
  ops: [matmul]
  blocks: [[1, 1]]
analyses: [compare_cases, energy_by_engine, energy_per_flop_by_engine, pj_per_flop_by_engine, pj_per_flop]
```

The four `long_matmul` configs under `sweeps/` set `mode: legacy` so they recreate the published
results; see `long_matmul`'s README for what `updated` changes. Leave `mode` out for new
measurements.

Analysis steps: `compare_cases` (grouped bar charts per case), `energy_by_engine` and
`energy_per_flop_by_engine` (reader/writer/compute ablation against `writer_amp`, average
charge, `writer_idle` as the writer case), `pj_per_flop_by_engine` (the published Fig. 1: peak
charge, `regular` as the writer stand-in), `cross_op` and `cross_op_corrected` (ops side by side,
shared or per-op FLOP denominator), `pj_per_flop`, `naive_vs_blocked`, `compare_runs`,
`op_breakdown`. Options go in a nested mapping, e.g.
`- naive_vs_blocked: {naive: matmul/regular, blocked: matmul/regular_b2x4, board: "Blackhole p100a"}`.
The pJ/FLOP steps need `long_matmul` in split mode: with `fixed_blocks_per_core` set the FLOP
count depends on the grid, so they are skipped.

### `compare_runs.py` — Cross-Run Comparison

Scans an output root directory for all `program_intervals.csv` files produced by `parser.py`, overlays the same metrics across runs on shared axes, and saves comparison figures to a separate output directory.

```bash
# Compare all runs
python3 compare_runs.py --out-root ./out --output-dir ./out/comparison

# Compare only specific runs
python3 compare_runs.py --out-root ./out --filter prefill_ fixed_tpc_ --output-dir ./out/comparison
```

### `compare_cases.py` — Named Use-Case Comparison

Compares a small, explicitly-named set of runs (e.g. the `POWER_CASE` power-experiment
scenarios — see [`long_matmul`'s README](../../tt_metal/programming_examples/long_matmul/README.md#power-experiment-knobs))
against each other, grouped by grid (core combination), as grouped bar charts. Unlike
`compare_runs.py` — which auto-discovers and overlays *all* runs under a root as line plots —
`compare_cases.py` takes an explicit, ordered list of run subdirectories from a small text
file, and renders one bar per use case per grid so cases are easy to read side by side.

Each named run must be a subdirectory of `--input-dir` containing a `program_intervals.csv`
(i.e. the output of a `parser.py` run with `--program-log`).

The cases file maps `Case N: subdir_name` to a run subdirectory, one per line, with an optional
third field for a custom legend label (defaults to `subdir_name` if omitted):

```
Case 1: compute_idle : Compute disabled
Case 2: reader_idle2 : Reader Idle
Case 3: writer_idle : Writer idle
Case 4: writer_amp : W+R+C active
```

```bash
python3 compare_cases.py -i ./out_new -c ./cases.txt
```

Outputs, written to `<input-dir>/compare_runs_out/`:

| File | Metric (from `program_intervals.csv`) |
|---|---|
| `dynamic_consumption_avg.png` | `dynamic_charge_avg_mah` |
| `dynamic_consumption_peak.png` | `dynamic_charge_peak_mah` |
| `dynamic_current_avg.png` | `dynamic_avg_current_a` |
| `dynamic_current_peak.png` | `dynamic_peak_current_a` |
| `execution_time.png` | `algo_time_s_reported` |
| `total_consumption.png` | `charge_avg_mah` |

Each chart groups grids (sorted by ascending core count) on the x-axis, with one bar per use
case per grid. Grids missing from a given case's CSV are left as a gap — with a warning printed
to stderr — rather than failing the whole comparison.

### `POWER_CASE` scenarios

The `POWER_CASE` sweep (`sweeps/paper_fig1_stage_breakdown.yaml`, `sweeps/crossop.yaml`) runs
`auto.py` once per case with `POWER_CASE` and `LONG_MATMUL_OP` exported into the application's
environment and `--subdir` set to the case's canonical name (`writer_amp`, `compute_idle`,
`reader_idle2`, `writer_idle`, `regular`), then `compare_cases.py` and the per-engine ablation
charts in `analysis/power_cases.py` against them. The cross-op pJ/FLOP charts divide every op by
matmul's `2*M*N*K`, so they compare energy at equal data movement; `cross_op_corrected` uses
each op's own FLOP count instead. See
[`long_matmul`'s README](../../tt_metal/programming_examples/long_matmul/README.md#power_case--the-six-canonical-scenarios)
for what each `POWER_CASE` actually disables or amplifies in the kernels.

## Output Structure

Each run produces a self-contained subdirectory:

```
output_root/subdir/
  telemetry.txt                    raw telemetry log
  summary.txt                      application stdout
  launcher.log                     auto.py orchestration log
  run_complete.json                written last, once the run succeeded; its settings
  program_intervals.csv            per-interval metrics (29 columns)
  Figures/
    telemetry_overview.png         3-panel time series: VCORE / TDC / TDP
    duration_vs_avg_current.png
    duration_vs_peak_current.png
    duration_vs_dynamic_avg_current.png
    duration_vs_dynamic_peak_current.png
    charge_avg_current.png
    charge_peak_current.png
    dynamic_charge_avg_current.png
    dynamic_charge_peak_current.png
    q_compute_vs_cores.png
    base_current.png
    charge_total_vs_compute_avg.png
    charge_total_vs_compute_peak.png
  profile_log_device.csv           (profiler runs only) copy of the device profiler CSV
  kernel_intervals.csv             (profiler runs only)
  kernel_utilization.csv           (profiler runs only)
  kernel_figures/                  (profiler runs only) see docs/device-side-instrumentation.md
```

## The Benchmark Application

Matrix multiplication is the fundamental building block of every modern AI workload. Transformer attention mechanisms, feed-forward layers, convolutional networks, and embedding lookups all reduce, at their core, to the operation **C = A × B**. This makes a sustained, configurable matrix multiply a good synthetic workload for power analysis — it exercises the same hardware paths (the matrix and vector engines, DRAM bandwidth, on-chip SRAM, the NoC) that a production LLM inference or training workload would. The workload is `tt_metal/programming_examples/long_matmul/`; its README documents every knob.

The application runs a sequence of matrix multiplications across a sweep of compute grid configurations generated from the device's compute grid, from 3×2 up to the full grid. For each grid size it allocates the input matrices A (M×K) and B (K×N) in DRAM once, then repeatedly executes the matmul for `num_iterations` iterations, measuring wall-clock time and reporting throughput in TFLOPS. Between grid configurations the application inserts a 5-second idle pause, which is precisely the quiet window that `parser.py` uses to estimate the idle power baseline.

### Arguments

```
metal_example_long_matmul  M  N  K  num_iterations  [fixed_blocks_per_core]
```

The first four arguments define the matrix dimensions and workload intensity. **M** is the number of rows of matrix A and the output matrix C, **N** is the number of columns of matrix B and the output C, and **K** is the shared inner dimension. All three must be divisible by 32, since the hardware processes data exclusively in 32×32 BFloat16 tiles. **`num_iterations`** controls how many times the full matmul is repeated per grid configuration, directly setting the duration of each measurement window.

The optional fifth argument, **`fixed_blocks_per_core`**, selects the operating mode. When omitted or set to zero, the application runs in **split mode**: total output tiles are divided equally across all active cores, so adding more cores reduces per-core work while keeping total computation constant — useful for measuring execution time scaling and efficiency. When set to a positive integer, every core computes exactly that many output blocks (single tiles unless `LONG_MATMUL_BLOCK_M/N` are set) per iteration regardless of grid size (**fixed-per-core mode**), meaning total work and power scale linearly with core count — useful for exposing how power delivery scales with the number of active compute units.

## Power Metrics

### Synchronization

The correlation between power measurements and program intervals relies entirely on wall-clock timestamp alignment. The telemetry binary runs continuously, writing timestamped samples to `telemetry.txt`. The application writes start and end timestamps for each grid configuration to its stdout, captured as `summary.txt`. Both use the same system clock. `parser.py` performs a post-hoc join: for each program interval it selects all telemetry samples whose timestamps fall within `[start_time, end_time]`. A configurable trim margin (`--trim-ms`, default 6 ms) is applied inward from both edges to discard the ramping-in and ramping-out transients, and then up to `--settle-max-ms` (default 25 ms, at most 10% of the window) of start-up spike is dropped, as reported in `settle_trim_ms`. Always pass `--app-args M N K iterations` to the app: with no arguments `long_matmul` does a one-grid smoke run, which has no idle pause to measure the base current in.

### Metric Definitions

**Base current** (`base_current_a`) is the mean TDC measured during the 5-second idle pause after the first program interval, used as one fixed reference for every interval of the run. It represents the irreducible power floor of the system — memory retention, voltage regulator overhead, clocking infrastructure, and thermal management — and is used to isolate the compute contribution from all other metrics.

**Duration** (`window_s`) is the wall-clock length of the trimmed measurement window. `sample_count` records how many telemetry samples fall within it and serves as a quality indicator.

**Average current** (`avg_current_a`) is the arithmetic mean of all TDC samples in the window, representing the typical sustained load. **Peak current** (`peak_current_a`) is the maximum observed TDC value, relevant for power delivery dimensioning and protection circuitry ratings.

**Dynamic average current** (`dynamic_avg_current_a`) and **dynamic peak current** (`dynamic_peak_current_a`) subtract the base current from their raw counterparts, isolating the current attributable exclusively to computation. In the default `updated` mode the sign is kept, so a case that draws less than the idle baseline shows a negative value; `legacy` floors it at zero.

**Total charge** (`charge_avg_mah`, `charge_peak_mah`) integrates average and peak current over the window duration, converted to milliampere-hours. These capture the full device consumption including baseline. **Dynamic charge** (`dynamic_charge_avg_mah`, `dynamic_charge_peak_mah`) applies the same integration to the baseline-subtracted currents, representing charge consumed purely by the computation. The figure `q_compute_vs_cores` plots dynamic peak charge across the grid sweep against an ideal linear scaling reference. The reference assumes work grows with the core count, so it is drawn only in fixed-per-core mode (the app prints `Mode: FIXED`); there, deviation from it points to bandwidth saturation, interconnect contention, or thermal throttling.

**Energy** is computed by two independent methods. TDP-based energy (`energy_j_tdp`, `energy_mwh_tdp`) uses the power value reported directly by the hardware sensor. V×I energy (`energy_j_vi`, `energy_mwh_vi`) computes instantaneous power from measured voltage and current per sample. The TDP signal is internally filtered and may be quantized by firmware; V×I is physically more precise but more susceptible to combined sensor noise. A consistent gap between the two across grid sizes can indicate a calibration offset in one of the sensors.

### Device-Side Kernel Instrumentation

An optional layer correlates per-kernel (reader / compute / writer) device-side timestamps with the same power telemetry, producing a per-core wait-vs-active-vs-overhead breakdown. See [`docs/device-side-instrumentation.md`](docs/device-side-instrumentation.md).
