# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

tt-ember collects and analyzes power metrics (voltage, current, power) from Tenstorrent hardware (TT-Metal devices) during algorithm execution. It produces per-interval statistics, charge/energy calculations, and matplotlib visualizations.

## Repository Structure

```
tt-ember/
  auto.py              Orchestrator — runs telemetry + app + parser in sequence
  parser.py            Analysis engine — metrics, CSV, figures
  run_all.py           Sweep runner — calls auto.py for 6 predefined configurations
  compare_runs.py      Cross-run comparison — overlays metrics across runs
  compare_runs2.py     Named use-case comparison — grouped bar charts per grid
  run_power_cases.py   POWER_CASE sweep runner — reset + auto.py per case + compare
  README.md            Full project documentation
  .gitignore           Ignores out/, out2/, out3/, Old/, Results/
  diagnostics/
    test.py            Inspect tt_umd API structure
    test2.py           Direct device polling via tt_umd
    test3.py           Live real-time power visualization
    tt_telem_diag.py   Periodic polling via tt-smi JSON snapshots
```

## Environment Requirements

- Activate a Python environment with tt-metal's Python dependencies before running anything: `source /path/to/tt-metal-venv/bin/activate`
- Point `$TT_METAL_HOME` at your `tt-metal` checkout: `export TT_METAL_HOME=/path/to/tt-metal`
- The following binaries must be built from the `tt-metal` repo before running the pipeline:
  - `$TT_METAL_HOME/build_Release/tools/umd/telemetry`
  - `$TT_METAL_HOME/build_Release/programming_examples/metal_example_high_power_matmul`
- `tt_umd` library required for `diagnostics/test.py`, `diagnostics/test2.py`, `diagnostics/test3.py`
- `numpy`, `matplotlib`, `pandas` required (`parser.py`, `compare_runs.py`, `compare_runs2.py`)

## Running the Tools

### Full sweep — 6 runs (3 prefill + 3 fixed-tpc) with auto reset and comparison

`run_all.py` is the top-level entry point. Before each run it issues `tt-smi -r` and waits 10s. It skips runs and comparisons whose output directories already exist. After all runs it invokes `compare_runs.py` twice (prefill and fixed-tpc separately).

```bash
python3 run_all.py \
  --telemetry-exe $TT_METAL_HOME/build_Release/tools/umd/telemetry \
  --app-exe $TT_METAL_HOME/build_Release/programming_examples/metal_example_high_power_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root $TT_METAL_HOME \
  --output-root ./out
```

### Single run — orchestrates telemetry + app + parser

**IMPORTANT:** `--app-args` must always be **last** — it uses `argparse.REMAINDER` and will consume every argument that follows it.

```bash
python3 auto.py \
  --telemetry-exe $TT_METAL_HOME/build_Release/tools/umd/telemetry \
  --telemetry-freq 50 \
  --app-exe $TT_METAL_HOME/build_Release/programming_examples/metal_example_high_power_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root $TT_METAL_HOME \
  --output-root ./out \
  --subdir prefill_4096_8192_8192_160 \
  --slot-ms 1 \
  --device-id 0 \
  --trim-ms 6.0 \
  --app-args 4096 8192 8192 160
```

### Parser standalone — analyze an existing telemetry capture

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

### Cross-run comparison

```bash
python3 compare_runs.py --out-root ./out --output-dir ./out/comparison_prefill --filter prefill
python3 compare_runs.py --out-root ./out --output-dir ./out/comparison_fixed_tpc --filter fixed_tpc
```

### Diagnostic utilities

```bash
python3 diagnostics/test2.py --interval 0.1          # live polling via tt_umd
python3 diagnostics/test3.py --interval 0.5           # live matplotlib visualization
python3 diagnostics/tt_telem_diag.py --interval 1.0 --output power_log.csv
python3 diagnostics/test.py                           # inspect tt_umd API
```

## Architecture

### Data Flow

```
run_all.py (sweep orchestrator)
  ├── tt-smi -r (hardware reset before each run)
  ├── auto.py × 6 (3 prefill + 3 fixed-tpc)
  └── compare_runs.py × 2 (comparison_prefill/, comparison_fixed_tpc/)

auto.py (single-run orchestrator)
  ├── starts telemetry exe (background subprocess)
  ├── runs TT-Metal app (foreground, captures stdout to summary.txt)
  ├── waits 10s cool-down, then kills telemetry
  └── invokes parser.py on telemetry.txt + summary.txt

parser.py (analysis engine)
  ├── parse_file_binned()        → 1ms time-slot bins, keyed by (device_id, slot_ms)
  ├── build_timeseries()         → numpy arrays per device
  ├── parse_program_log()        → list of ProgramInterval (from summary.txt)
  ├── compute_interval_metrics() → per-interval stats with dynamic baseline
  ├── write CSV (29 columns)
  └── plot_png() + plot_program_interval_figures() → 14+ PNG charts
```

### Key Modules

**[parser.py](parser.py)** — Core analysis. All metric computation and visualization lives here.
- `Sample` (frozen dataclass): `(ts, device_id, vcore_mv, tdc_a, tdp_w)`
- `ProgramInterval` (dataclass): execution window with grid size, core count, timing, and computed metrics fields
- Regex constants `RE_TS`, `RE_DEV`, `RE_METRIC`, `RE_PROGRAM_ROW` drive all log parsing
- `compute_interval_metrics()` does dynamic baseline subtraction: idle power measured from pause gaps between intervals is subtracted to isolate compute-only current

**[auto.py](auto.py)** — Orchestration only. Subprocess lifecycle, output directory creation, and pipeline wiring. No metric math here.
- Process shutdown: SIGTERM → 10s wait → SIGKILL
- `launcher.log` records all subprocess commands and the full interval analysis table — use it to reconstruct any past invocation

**[run_all.py](run_all.py)** — Sweep orchestrator. Defines 6 predefined `RunSpec` entries (3 prefill, 3 fixed-tpc). Smart resume: checks each run dir and each comparison dir independently — skips what exists, runs what doesn't. Runs `compare_runs.py` after simulations complete.

**[compare_runs.py](compare_runs.py)** — Loads `program_intervals.csv` from multiple run subdirs and overlays metrics across runs. Produces overcompute, percentage-change, and per-metric comparison figures.

**[diagnostics/test2.py](diagnostics/test2.py)** — Architecture-aware direct polling via `tt_umd`. Handles Wormhole B0 and Blackhole separately.

**[diagnostics/tt_telem_diag.py](diagnostics/tt_telem_diag.py)** — Polls `tt-smi -s` JSON output. Resilient to multiple JSON schema variants.

### Predefined Runs in run_all.py

**Prefill** (matrix size sweep, split mode):
- `prefill_1024_2048_2048_160` — app args: `1024 2048 2048 160`
- `prefill_2048_4096_4096_160` — app args: `2048 4096 4096 160`
- `prefill_4096_8192_8192_160` — app args: `4096 8192 8192 160`

**Fixed tiles-per-core** (fixed-per-core mode, power scales with core count):
- `fixed_tpc_100` — app args: `2048 4096 4096 160 100`
- `fixed_tpc_200` — app args: `2048 4096 4096 160 200`
- `fixed_tpc_400` — app args: `2048 4096 4096 160 400`

### Output Structure per Run

```
output_root/subdir/
  telemetry.txt                  raw telemetry log (input copy)
  summary.txt                    application stdout (input copy)
  launcher.log                   full orchestration log + interval analysis table
  program_intervals.csv          per-interval metrics (29 columns)
  Figures/
    telemetry_overview.png       3-panel time series: VCORE / TDC / TDP
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
    q_compute_vs_cores.png        dynamic peak charge vs cores with ideal linear scaling
    base_current.png
    charge_total_vs_compute_avg.png
    charge_total_vs_compute_peak.png

output_root/comparison_prefill/   generated by run_all.py after prefill runs
output_root/comparison_fixed_tpc/ generated by run_all.py after fixed-tpc runs
```

### Metrics Computed per Interval (29 CSV columns)

- `window_s` — trimmed interval duration
- `avg_current_a`, `peak_current_a` — raw TDC statistics
- `avg_vcore_v`, `avg_tdp_w`, `avg_power_vi_w` — voltage and power averages
- `charge_avg_mah`, `charge_peak_mah` — total charge (mAh)
- `energy_j_tdp`, `energy_mwh_tdp` — TDP-based energy
- `energy_j_vi`, `energy_mwh_vi` — V×I energy
- `base_current_a` — idle current estimated from pauses between intervals
- `dynamic_avg_current_a`, `dynamic_peak_current_a` — after subtracting `base_current_a`
- `dynamic_charge_avg_mah`, `dynamic_charge_peak_mah` — compute-only charge
- `base_pause_left_samples`, `base_pause_right_samples` — baseline quality indicators

### Known Gotchas

- `--app-args` in `auto.py` uses `argparse.REMAINDER` — it must always be the **last** argument or it will silently consume all subsequent arguments, causing a confusing "required argument missing" error.
- Telemetry binary path is `build_Release` not `build` — double-check when building.
- Large telemetry `.txt` files (100MB+) must not be committed to git — `out/`, `Old/`, `Results/` are gitignored.
- To recover a past `auto.py` invocation, read `launcher.log` in the run directory — it contains the full subprocess commands and interval analysis table.

### Telemetry Log Format

```
2026-05-06 10:17:54.470123 Device ID 0 Chip ID 0: TDP: 31.2 W, TDC: 28.3 A, VCORE: 847 mV
```

Summary log row format (parsed by `RE_PROGRAM_ROW`):
```
3x2   6   238.736697   0.54   1.23   2026-05-06 10:17:58.112000   2026-05-06 10:21:56.632200
```
Fields: `grid  cores  algo_time_s  tflops  per_iter_ms  start_time  end_time`
