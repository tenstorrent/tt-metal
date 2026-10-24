# tt-ember — Device-Side Kernel Instrumentation

This document covers the optional kernel-level layer on top of the core tt-ember pipeline
(`auto.py` + `parser.py` + telemetry): correlating `long_matmul`'s reader/compute/writer
kernel activity with the voltage/current/power telemetry. For the core telemetry-only pipeline,
see [`README.md`](../README.md).

## 1. Overview

With the device profiler on, `long_matmul`'s reader/compute/writer kernels record device-side
timestamps (`DeviceTimestampedData`), and `parser.py` converts those into host wall-clock time
and joins them with the same run's telemetry — producing a per-core, per-kernel **wait vs.
active vs. overhead** breakdown, plus charts in `kernel_figures/`.

## 2. Pipeline overview

```
auto.py (orchestrator)
  ├── starts telemetry exe (background)
  ├── deletes any stale generated/profiler/.logs/profile_log_device.csv
  ├── runs long_matmul (foreground) -- app emits device profiler markers if
  │   TT_METAL_DEVICE_PROFILER=1 is set
  ├── waits, stops telemetry
  └── invokes parser.py, passing --device-profiler-csv if the file exists

parser.py (analysis engine)
  ├── telemetry.txt + summary.txt -> program_intervals.csv + Figures/*.png
  └── with --device-profiler-csv: + profile_log_device.csv -> kernel_intervals.csv,
      kernel_utilization.csv, kernel_figures/*.png
```

Without `--device-profiler-csv` (or when the app didn't produce one), only the telemetry outputs
are written.

## 3. Device-side markers (what the kernels emit)

Each kernel marks its own start/end plus an aggregate wait-vs-active cycle count, once per
invocation (not per iteration):

| Kernel  | Markers | "Wait" = blocked in... | "Active" = time in... |
|---------|---------|-------------------------|------------------------|
| Reader  | `READER_KERNEL_START/END`, `READER_WAIT_CYCLES`, `READER_TRANSFER_CYCLES` | `cb_reserve_back` (input CB full) | `noc_async_read_page` + barrier |
| Compute | `COMPUTE_KERNEL_START/END`, `COMPUTE_WAIT_CYCLES`, `COMPUTE_COMPUTE_CYCLES` (once **per TRISC** — 3x per core) | TRISC0: `cb_wait_front` (no input yet); TRISC1: `tile_regs_acquire` (DEST still being packed); TRISC2: `tile_regs_wait` + `cb_reserve_back` (math not done, output CB full) | TRISC1: the math loop; TRISC2: `pack_tile`; TRISC0: the unpacker blocking inside `matmul_tiles` / `add_tiles` / `copy_tile` |
| Writer  | `WRITER_KERNEL_START/END`, `WRITER_WAIT_CYCLES`, `WRITER_TRANSFER_CYCLES` | `cb_wait_front` (no output tile yet) | `noc_async_write_page` + barrier |

Source: `tt_metal/programming_examples/long_matmul/kernels/{dataflow,compute}/*.cpp` and
`kernels/profile_markers.h`. The host (`long_matmul.cpp`) drains these via
`ReadMeshDeviceProfilerResults(*mesh_device)` after every grid run, into
`generated/profiler/.logs/profile_log_device.csv`. The timestamp reads are compiled in only when
the profiler is on, so power runs without it execute the same instructions.

**Compute runs on three RISC-V processors in each Tensix core** (TRISC0=unpack, TRISC1=math,
TRISC2=pack); the reader and writer run on the core's two data-movement RISCs. The *same* compute
source is compiled 3 times, and tt-metal's own profiler tags each marker with which
one recorded it (the `RISC processor type` column). Only TRISC1 calls `matmul_tiles`; the three
roles have very different wait/active profiles and must **not** be averaged together (see §6).

**To capture the markers**, set `TT_METAL_DEVICE_PROFILER=1`. `parser.py` converts device cycles
to wall-clock time itself, so `TT_METAL_PROFILER_SYNC` is not needed: each grid's first marker is
pinned to the grid's printed start time, and cycles are converted with the profiler CSV's
`CHIP_FREQ` (in `legacy` mode, with a rate fitted from the grid's printed start/end times instead). Without the profiler the app runs as usual and writes no
`profile_log_device.csv`; `auto.py` then skips the kernel-level analysis.

Keep profiler runs and power runs separate. With the profiler on, the kernels read the timer in
their inner loops and the profiler is read back during the first idle pause, which is where the
base current is measured. Unset `TT_METAL_DEVICE_PROFILER` before a power sweep
(`run_power_cases.py` and `run_all.py` pass the environment through).

## 4. `parser.py`

### 4.1 CLI arguments

| Argument | Meaning |
|---|---|
| `--device-profiler-csv PATH` | Path to `profile_log_device.csv`. Requires `--program-log`; enables everything in this document. |
| `--kernel-csv PATH` | Optional override for the collapsed per-kernel CSV output path (default `<output>/<subdir>/kernel_intervals.csv`). |

The other arguments (`-i`, `-o`, `--subdir`, `--program-log`, `--slot-ms`, `--trim-ms`,
`--settle-max-ms`, `--device-id`, `--dpi`, `--program-csv`) are the same as for the telemetry-only analysis.

### 4.2 Outputs

Written to `<output-root>/<subdir>/` alongside the existing `program_intervals.csv`/`Figures/`:

- **`profile_log_device.csv`** — a copy of the input device-profiler CSV (for reproducibility, same pattern as the existing telemetry/summary copies).
- **`kernel_intervals.csv`** — one row per `(grid, kernel)`, collapsed across all cores of that kernel type: `grid, cores, kernel, start_time, end_time, duration_s`. This is the coarse, whole-kernel view.
- **`kernel_utilization.csv`** — the main output: one row per `(grid, kernel, core_x, core_y, risc)`:

  | Column | Meaning |
  |---|---|
  | `kernel_start`, `kernel_end` | That core's own start/end, converted to host wall-clock time |
  | `lifetime_s` | `kernel_end - kernel_start` |
  | `wait_s` / `wait_pct` | From `*_WAIT_CYCLES`, converted to seconds / % of `lifetime_s` |
  | `active_s` / `active_pct` | From `*_TRANSFER_CYCLES` or `*_COMPUTE_CYCLES` |
  | `overhead_s` / `overhead_pct` | **Derived**, not measured: `max(0, lifetime_s - wait_s - active_s)` — see §7 |

- **`kernel_figures/`**:
  - `reader_utilization.png`, `writer_utilization.png` — stacked bar (active/overhead/wait %) vs. grid size, averaged over cores.
  - `compute_unpack_utilization.png`, `compute_math_utilization.png`, `compute_pack_utilization.png` — same, but **one file per TRISC role** (never blended — see §6).
  - `combined_utilization_pct.png` / `combined_utilization_seconds.png` — Reader / Compute(TRISC1-math) / Writer **side by side** on one chart, active+overhead+wait stacked. The seconds version adds a 4th "Run duration" reference bar per grid (§8).

### 4.3 Key functions

| Function | Purpose |
|---|---|
| `load_device_kernel_markers()` | Parses `profile_log_device.csv`. Note: `skipinitialspace=True` is required — the CSV header uses `", "` separators, which otherwise makes every column key have a leading space. |
| `check_marker_counts()` | Checks that every run recorded `cores x 1` reader/writer and `cores x 3` compute start/end markers. If not, the kernel analysis is skipped and the run is marked incomplete. |
| `compute_run_windows()` | Gives each run its markers by count, in cycle order (`cores x 1` or `x 3` per kernel, with each run's core count from the program log). Anchors the run at the *global* min(`*_KERNEL_START`) across all 3 kernels, pinned to the run's start time, and returns its cycle window and `freq_hz`: `CHIP_FREQ` from the profiler CSV, or in `legacy` mode the cycle span divided by the host start-to-end time. Core coordinates are not used — `core_x`/`core_y` in the CSV are physical NOC coordinates, not the logical grid coordinates used to launch kernels. |
| `find_run_index()` | Assigns any other marker to the run whose cycle window contains it. |
| `compute_kernel_intervals()` | Collapsed per-kernel view → `kernel_intervals.csv`. |
| `compute_kernel_utilization_rows()` | Per-core view → `kernel_utilization.csv`, including the derived-overhead calculation and the sanity clamp described in §7. |
| `aggregate_kernel_utilization()` / `plot_kernel_utilization()` | Per-`(grid, kernel, risc)` averages and the 5 individual charts. |
| `aggregate_combined_utilization()` / `plot_combined_utilization()` | The 2 combined charts, including the "Run duration" reference bar logic. |

## 5. `auto.py`

- `--device-profiler-csv PATH` argument (default: `$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv`, or `<tt-metal-root>/generated/profiler/.logs/profile_log_device.csv`, which is where tt-metal writes it).
- **Deletes** that file (if present) immediately before starting the app, so it only ever contains this run's fresh data (tt-metal itself also resets it on first `DeviceProfiler` construction, but this is a second layer of safety against stale data from a previous manual run).
- After the app finishes, checks whether the file now exists; if so, passes it to `parser.py` via `--device-profiler-csv`. If not (e.g. `TT_METAL_DEVICE_PROFILER` wasn't set), logs a note and skips it — the rest of the pipeline still runs normally.

## 6. Why compute has 3 separate charts, and why they can't be summed

`compute.cpp` runs independently on the three TRISCs of each core. Each has its own 100% (its own lifetime), and they overlap in wall-clock time
rather than partitioning it, so their percentages cannot be summed. Measured on Wormhole n150,
`POWER_CASE=1`, `1024 2048 2048 20`, 1x1 blocking, grids 3x2 to 8x8:

| Role | Wait % | Active % | What it's actually doing |
|---|---|---|---|
| TRISC0 (unpack) | 92–97 | 1–3 | Feeds tiles from L1 into the FPU's source registers |
| TRISC1 (math) | 1–2 | 88–94 | The only role that issues `matmul_tiles` |
| TRISC2 (pack) | 87–98 | ~0.1 | Packs FPU results back into the output CB; mostly waits for the writer to free it |

Same logic applies across kernels: reader/compute/writer run **concurrently** on separate
processors for virtually the entire grid run (tightly coupled via the depth-2 circular buffers), so
their wait/active percentages are each "% of *that kernel's own* lifetime" — not one shared
100% budget.

## 7. Why "overhead" is *derived*, not measured — and what it really means

Measuring a third "overhead" bucket on the device (wrapping
`tile_regs_acquire/commit/wait/release`, `cb_pop_front`, index arithmetic, etc. in their own
timestamp pairs) was tried and read far too small even with explicit compiler barriers
(`asm volatile("" ::: "memory")`), because those spans have **no real hardware operation**
between the two timestamp reads — nothing forces the compiler to
keep surrounding (non-volatile) arithmetic from drifting relative to them, and cycle-counter
granularity compounds the error. `wait`/`active` were never affected by this, because they're
each bounded by a real hardware call (`cb_reserve_back`/`cb_wait_front` on one side,
`noc_async_read/write`/`matmul_tiles`/`pack_tile` on the other), which acts as a genuine
barrier.

**Fix: derive it instead.** `overhead_s = max(0, lifetime_s - wait_s - active_s)`. This is
exact by construction, since `lifetime_s` (from `KERNEL_START`/`KERNEL_END`) and `wait_s`/
`active_s` are all independently measured across real hardware operations.

Each TRISC times the blocking calls it actually executes (`cb_wait_front` only runs on TRISC0,
`tile_regs_acquire` on TRISC1, `tile_regs_wait` and `cb_reserve_back` on TRISC2), so the
writer's back-pressure shows up as TRISC2 wait rather than overhead. With
`LONG_MATMUL_DISABLE_COMPUTE=1` (§9) TRISC1 and TRISC2 have almost no active time (in
`updated` mode they still compute the first two blocks), and their overhead grows. "Overhead" is a catch-all for "everything not named wait or active," not a
claim about *why* that time is spent.

Two details worth knowing if you change this code:
- **Spans use the low 32 bits of the wall clock.** Reading both halves can tear, and a torn read
  added exactly `2^32` cycles to a span; `parser.py` then rejects the row (wait or active longer
  than the kernel's lifetime) and reports `nan`. Every span is far shorter than `2^32` cycles,
  so a 32-bit unsigned difference is exact.
- **CSV `skipinitialspace`**: the device CSV header has `", "` separators; without
  `skipinitialspace=True`, every `csv.DictReader` key except the first had a leading space,
  silently matching nothing.

## 8. The "Run duration" reference bar

`combined_utilization_seconds.png` adds a 4th bar per grid: the max `lifetime_s` across
Reader/Compute(TRISC1)/Writer for that grid (they're all expected to be equal, since the three
kernels start/finish together). Since reader/compute/writer run *concurrently*, none of their
own active+overhead+wait stacks should exceed this reference — if one does, something's wrong
(mismatched run assignment, a fresh measurement bug, etc.).

## 9. Power-experiment flags

`POWER_CASE`, `LONG_MATMUL_DISABLE_*`, write amplification, output blocking and
`LONG_MATMUL_OP` change what the kernels *do*; they are documented in
[`long_matmul`'s README](../../../tt_metal/programming_examples/long_matmul/README.md#power-experiment-knobs).
With `LONG_MATMUL_DISABLE_COMPUTE=1` the compute kernel's active time drops to near zero (in
`updated` mode it still computes the first two blocks) and its overhead grows (see §7).

## 10. Full example: one complete run with everything on

```bash
# Activate the Python environment that has tt-metal's Python deps (tt_umd, etc.)
source /path/to/tt-metal-venv/bin/activate

# Point this at your tt-metal checkout
export TT_METAL_HOME=/path/to/tt-metal

export TT_METAL_DEVICE_PROFILER=1
export POWER_CASE=1        # or any of the individual LONG_MATMUL_* flags instead

python3 auto.py \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --telemetry-freq 50 \
  --app-exe "$TT_METAL_HOME"/build_Release/programming_examples/metal_example_long_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out \
  --subdir power_case_1_test \
  --slot-ms 1 \
  --device-id 0 \
  --trim-ms 1.0 \
  --app-args 1024 2048 2048 160
```

`--app-args` must be last (it consumes everything after it). Results land in
`out/power_case_1_test/`: `program_intervals.csv` + `Figures/` (existing telemetry analysis),
`kernel_intervals.csv` + `kernel_utilization.csv` + `kernel_figures/` (device-side analysis).

### Re-running just the parser (no hardware needed)

If you already have `telemetry.txt` + `summary.txt` + `profile_log_device.csv` from a previous
run (e.g. you only changed `parser.py`, not the kernels/app), you don't need to re-run the
hardware — just re-invoke the parser directly against the existing files, writing back into
the same run directory. This removes the run's `run_complete.json`, since its settings no longer
describe the new results, so a later sweep into the same output root measures that run again.

```bash
python3 parser.py \
  -i out/power_case_1_test/telemetry.txt \
  --program-log out/power_case_1_test/summary.txt \
  -o out \
  --subdir power_case_1_test \
  --slot-ms 1 --device-id 0 --trim-ms 1.0 \
  --device-profiler-csv out/power_case_1_test/profile_log_device.csv
```
