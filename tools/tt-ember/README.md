<div align="center">

<h1>

[Hardware](https://tenstorrent.com/cards/) | [Documentation](https://docs.tenstorrent.com/) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://job-boards.greenhouse.io/tenstorrent?gh_src=22e462047us)

</h1>

<picture>
  <img alt="tt-ember logo" src="docs/images/tt_ember_logo.png" height="220">
</picture>

**Power and energy telemetry for Tenstorrent hardware.**

</div>
<br>

tt-ember collects and analyzes power metrics (voltage, current, power) from Tenstorrent hardware during workload execution, built on [TT-Metalium](https://github.com/tenstorrent/tt-metal). It produces per-interval statistics, charge and energy calculations, and matplotlib visualizations — all aligned to the execution windows of the measured application. An optional device-side layer correlates per-kernel activity (reader / compute / writer) with the same power telemetry for a wait-vs-active-vs-overhead breakdown.

-----
**Contents:** [Overview](#overview) · [Repository Structure](#repository-structure) · [Prerequisites](#prerequisites) · [Quick Start](#quick-start) · [Scripts](#scripts) · [Output](#output-structure) · [Benchmark App](#the-benchmark-application) · [Power Metrics](#power-metrics) · [Contributing](#contributing) · [License](#license)

-----

## Overview

tt-ember runs a measured application (by default a high-power matmul benchmark) while a telemetry binary samples the device's power sensors, then joins the two by wall-clock time to attribute power and energy to each phase of execution. Everything is driven by six Python scripts and a set of diagnostic utilities — no build step for the tooling itself; only the `tt-metal` binaries it drives need to be compiled.

## Repository Structure

```
tt-ember/
  auto.py              Orchestrator — runs telemetry + app + parser in sequence
  parser.py            Analysis engine — metrics, CSV, figures
  run_all.py           Sweep runner — calls auto.py for multiple configurations
  compare_runs.py      Cross-run comparison — overlays metrics across runs
  compare_runs2.py     Named use-case comparison — grouped bar charts per grid
  run_power_cases.py   POWER_CASE sweep runner — reset + auto.py per case + compare
  diagnostics/
    test.py            Inspect tt_umd API structure
    test2.py           Direct device polling via tt_umd
    test3.py           Live real-time power visualization
    tt_telem_diag.py   Periodic polling via tt-smi JSON snapshots
  docs/
    device-side-instrumentation.md   Per-kernel (reader/compute/writer) analysis
```

## Prerequisites

### Python environment

Activate a Python environment that has tt-metal's Python dependencies (`tt_umd`, etc.) available:

```bash
source /path/to/tt-metal-venv/bin/activate
```

### Built binaries

The following binaries must be compiled from the [`tt-metal`](https://github.com/tenstorrent/tt-metal) repository before running the pipeline. Point `$TT_METAL_HOME` at your checkout:

```bash
export TT_METAL_HOME=/path/to/tt-metal
```

```
$TT_METAL_HOME/build_Release/tools/umd/telemetry
$TT_METAL_HOME/build_Release/programming_examples/metal_example_high_power_matmul
```

## Quick Start

Run one full measurement — telemetry capture, application, and analysis — in a single command:

```bash
python3 auto.py \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --telemetry-freq 50 \
  --app-exe "$TT_METAL_HOME"/build_Release/programming_examples/metal_example_high_power_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out \
  --subdir prefill_4096_8192_8192_160 \
  --slot-ms 1 \
  --device-id 0 \
  --trim-ms 6.0 \
  --app-args 4096 8192 8192 160
```

> `--app-args` must always be placed **last** — it uses `argparse.REMAINDER` and will consume every argument that follows it.

Results land in `./out/prefill_4096_8192_8192_160/` (see [Output Structure](#output-structure)).

## Scripts

### `auto.py` — Full Pipeline Orchestrator

Manages the full measurement lifecycle: starts the telemetry binary as a background subprocess, runs the application in the foreground while capturing its stdout to `summary.txt`, waits for a 10-second cool-down, and invokes `parser.py` on the captured data. All output is written to a structured directory under `--output-root`.

### `parser.py` — Analysis Engine

Can be run standalone against a previously captured telemetry log. Parses raw voltage, current, and power samples, bins them into time slots, computes per-interval metrics, writes a 29-column CSV, and produces 14+ matplotlib figures.

```bash
python3 parser.py \
  -i ./out/prefill_4096_8192_8192_160/telemetry.txt \
  --program-log ./out/prefill_4096_8192_8192_160/summary.txt \
  -o ./out \
  --subdir prefill_4096_8192_8192_160 \
  --slot-ms 1 \
  --device-id 0 \
  --trim-ms 6.0
```

### `run_all.py` — Sweep Orchestrator

Runs a predefined sequence of measurements by invoking `auto.py` once per configuration. Before each run it issues a hardware reset via `tt-smi -r` and waits for the device to re-initialize, ensuring a clean thermal and electrical baseline. Stops immediately if any reset or run fails.

```bash
python3 run_all.py \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --app-exe "$TT_METAL_HOME"/build_Release/programming_examples/metal_example_high_power_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out
```

### `compare_runs.py` — Cross-Run Comparison

Scans an output root directory for all `program_intervals.csv` files produced by `parser.py`, overlays the same metrics across runs on shared axes, and saves comparison figures to a separate output directory.

```bash
# Compare all runs
python3 compare_runs.py --out-root ./out --output-dir ./out/comparison

# Compare only specific runs
python3 compare_runs.py --out-root ./out --filter prefill_2048 fixed_power_sweep --output-dir ./out/comparison
```

### `compare_runs2.py` — Named Use-Case Comparison

Compares a small, explicitly-named set of runs (e.g. the `POWER_CASE` power-experiment
scenarios — see [§9 of the device-side instrumentation doc](docs/device-side-instrumentation.md#9-power-experiment-flags-on-high_power_matmul-itself))
against each other, grouped by grid (core combination), as grouped bar charts. Unlike
`compare_runs.py` — which auto-discovers and overlays *all* runs under a root as line plots —
`compare_runs2.py` takes an explicit, ordered list of run subdirectories from a small text
file, and renders one bar per use case per grid so cases are easy to read side by side.

Each named run must be a subdirectory of `--input-dir` containing a `program_intervals.csv`
(i.e. the output of a `parser.py` run with `--program-log`).

The cases file maps `Case N: subdir_name` to a run subdirectory, one per line, with an optional
third field for a custom legend label (defaults to `subdir_name` if omitted):

```
Case 1: compute_idle : Compute disabled
Case 2: reader_idle2 : Reader Idle
Case 3: regular : Writer idle
Case 4: writer_amp : W+R+C active
```

```bash
python3 compare_runs2.py -i ./out_new -c ./cases.txt
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

### `run_power_cases.py` — POWER_CASE Sweep + Comparison

Automates the full `POWER_CASE` power-experiment workflow. For each requested `POWER_CASE`
value it resets the hardware (`tt-smi -r`), runs `auto.py` with `POWER_CASE` exported into the
application's environment and `--subdir` set to that case's canonical name, and — once every
case has finished — writes a `compare_runs2.py` cases file and calls it to produce the
comparison charts. This is the single-command equivalent of running the manual
reset/export/`auto.py` sequence once per case and then invoking `compare_runs2.py` by hand.

```bash
python3 run_power_cases.py \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --app-exe "$TT_METAL_HOME"/build_Release/programming_examples/metal_example_high_power_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out_new \
  --trim-ms 1.0 \
  --app-args 1024 2048 2048 160
```

Notes:
- `--app-args` must be last (same `argparse.REMAINDER` caveat as `auto.py`/`run_all.py`).
- Defaults to `POWER_CASE` values `0 1 2 4` — the four single-variable comparisons against
  baseline. Pass `--power-cases 0 1 2 3 4` to include case 3 (reader **and** compute both idle)
  as well.
- A case is skipped if `<output-root>/<subdir-for-that-case>` already exists; pass `--force` to
  re-run it anyway.
- Pass `--dry-run` to print every `tt-smi -r` / `auto.py` / `compare_runs2.py` command that
  would run, without touching the hardware — useful for checking the plan before committing to
  a multi-hour sweep.
- Writes `<output-root>/power_cases.txt` (overridable with `--cases-file-name`), then calls
  `compare_runs2.py -i <output-root> -c <output-root>/power_cases.txt`.

See [§9 of the device-side instrumentation doc](docs/device-side-instrumentation.md#9-power-experiment-flags-on-high_power_matmul-itself)
for what each `POWER_CASE` actually disables or amplifies in the kernels.

### Diagnostic Utilities (`diagnostics/`)

These scripts interact directly with the hardware outside of the main pipeline. They are useful for verifying device connectivity, inspecting live power behavior, or capturing a quick log without the full `auto.py` workflow.

- **`test2.py`** polls the device in real time via the `tt_umd` Python library and prints a timestamped current/voltage/power log to stdout. It handles both Wormhole B0 and Blackhole architectures automatically.
- **`test3.py`** does the same polling but renders the data as a live scrolling matplotlib plot.
- **`tt_telem_diag.py`** queries `tt-smi -s` JSON snapshots on a fixed interval and writes per-device power readings to a CSV file.
- **`test.py`** is a one-shot inspection tool that probes the `tt_umd` API structure and prints available device information.

```bash
python3 diagnostics/test2.py --interval 0.1
python3 diagnostics/test3.py --interval 0.5
python3 diagnostics/tt_telem_diag.py --interval 1.0 --output power_log.csv
python3 diagnostics/test.py
```

## Output Structure

Each run produces a self-contained subdirectory:

```
output_root/subdir/
  telemetry.txt                    raw telemetry log (input copy)
  summary.txt                      application stdout (input copy)
  launcher.log                     auto.py orchestration log
  program_intervals.csv            per-interval metrics (29 columns)
  Figures/
    telemetry_overview.png         3-panel time series: VCORE / TDC / TDP
    duration_vs_avg_current.png
    duration_vs_peak_current.png
    duration_vs_dynamic_avg_current.png
    duration_vs_dynamic_peak_current.png
    execution_time_vs_dynamic_avg_current.png
    execution_time_vs_dynamic_peak_current.png
    charge_avg_current.png
    charge_peak_current.png
    dynamic_charge_avg_current.png
    dynamic_charge_peak_current.png
    q_compute_vs_cores.png
    base_current.png
    charge_total_vs_compute_avg.png
    charge_total_vs_compute_peak.png
```

## The Benchmark Application

Matrix multiplication is the fundamental building block of every modern AI workload. Transformer attention mechanisms, feed-forward layers, convolutional networks, and embedding lookups all reduce, at their core, to the operation **C = A × B**. This makes a sustained, configurable matrix multiply kernel the ideal synthetic benchmark for power analysis — it exercises exactly the same hardware paths (tensor cores, DRAM bandwidth, on-chip SRAM, NoC interconnect) that a production LLM inference or training workload would.

The application runs a sequence of matrix multiplications across a predefined sweep of compute grid configurations, from a minimal 3×2 grid up to the full device grid. For each grid size it allocates the input matrices A (M×K) and B (K×N) in DRAM once, then repeatedly executes the matmul for `num_iterations` iterations, measuring wall-clock time and reporting throughput in TFLOPS. Between grid configurations the application inserts a 5-second idle pause, which is precisely the quiet window that `parser.py` uses to estimate the idle power baseline.

### Arguments

```
metal_example_high_power_matmul  M  N  K  num_iterations  [fixed_tiles_per_core]
```

The first four arguments define the matrix dimensions and workload intensity. **M** is the number of rows of matrix A and the output matrix C, **N** is the number of columns of matrix B and the output C, and **K** is the shared inner dimension. All three must be divisible by 32, since the hardware processes data exclusively in 32×32 BFloat16 tiles. **`num_iterations`** controls how many times the full matmul is repeated per grid configuration, directly setting the duration of each measurement window.

The optional fifth argument, **`fixed_tiles_per_core`**, selects the operating mode. When omitted or set to zero, the application runs in **split mode**: total output tiles are divided equally across all active cores, so adding more cores reduces per-core work while keeping total computation constant — useful for measuring execution time scaling and efficiency. When set to a positive integer, every core computes exactly that many tiles per iteration regardless of grid size (**fixed-per-core mode**), meaning total work and power scale linearly with core count — useful for exposing how power delivery scales with the number of active compute units.

## Power Metrics

### Synchronization

The correlation between power measurements and program intervals relies entirely on wall-clock timestamp alignment. The telemetry binary runs continuously, writing timestamped samples to `telemetry.txt`. The application writes start and end timestamps for each grid configuration to its stdout, captured as `summary.txt`. Both use the same system clock. `parser.py` performs a post-hoc join: for each program interval it selects all telemetry samples whose timestamps fall within `[start_time, end_time]`. A configurable trim margin (default 6 ms) is applied inward from both edges to discard the ramping-in and ramping-out transients, retaining only the steady-state active phase.

### Metric Definitions

**Base current** (`base_current_a`) is the mean TDC measured during the 5-second idle pause between consecutive program intervals. It represents the irreducible power floor of the system — memory retention, voltage regulator overhead, clocking infrastructure, and thermal management — and is used to isolate the compute contribution from all other metrics.

**Duration** (`window_s`) is the wall-clock length of the trimmed measurement window. `sample_count` records how many telemetry samples fall within it and serves as a quality indicator.

**Average current** (`avg_current_a`) is the arithmetic mean of all TDC samples in the window, representing the typical sustained load. **Peak current** (`peak_current_a`) is the maximum observed TDC value, relevant for power delivery dimensioning and protection circuitry ratings.

**Dynamic average current** (`dynamic_avg_current_a`) and **dynamic peak current** (`dynamic_peak_current_a`) subtract the base current from their raw counterparts, isolating the current attributable exclusively to computation. A floor of zero is enforced against measurement noise.

**Total charge** (`charge_avg_mah`, `charge_peak_mah`) integrates average and peak current over the window duration, converted to milliampere-hours. These capture the full device consumption including baseline. **Dynamic charge** (`dynamic_charge_avg_mah`, `dynamic_charge_peak_mah`) applies the same integration to the baseline-subtracted currents, representing charge consumed purely by the computation. The figure `q_compute_vs_cores` plots dynamic peak charge across the grid sweep against an ideal linear scaling reference — deviation from the reference indicates super- or sub-linear scaling, pointing to bandwidth saturation, interconnect contention, or thermal throttling.

**Energy** is computed by two independent methods. TDP-based energy (`energy_j_tdp`, `energy_mwh_tdp`) uses the power value reported directly by the hardware sensor. V×I energy (`energy_j_vi`, `energy_mwh_vi`) computes instantaneous power from measured voltage and current per sample. The TDP signal is internally filtered and may be quantized by firmware; V×I is physically more precise but more susceptible to combined sensor noise. A consistent gap between the two across grid sizes can indicate a calibration offset in one of the sensors.

### Device-Side Kernel Instrumentation

An optional layer correlates per-kernel (reader / compute / writer) device-side timestamps with the same power telemetry, producing a per-core wait-vs-active-vs-overhead breakdown. See [`docs/device-side-instrumentation.md`](docs/device-side-instrumentation.md).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](../../CONTRIBUTING.md) and our [Code of Conduct](../../CODE_OF_CONDUCT.md) before opening a pull request.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](../../LICENSE). See also [LICENSE_understanding.txt](../../LICENSE_understanding.txt).
