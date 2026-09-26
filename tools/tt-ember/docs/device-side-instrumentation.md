# tt-ember — Device-Side Kernel Instrumentation

This document covers everything added on top of the core tt-ember pipeline
(`auto.py` + `parser.py` + telemetry): correlating `high_power_matmul`'s reader/compute/writer
kernel activity with the existing voltage/current/power telemetry, and the power-experiment
flags added to `high_power_matmul` itself. For the core telemetry-only pipeline, see
[`README.md`](../README.md).

## 1. What's new, in one sentence

`high_power_matmul`'s reader/compute/writer kernels now emit device-side timestamps
(`DeviceTimestampedData`), and `parser.py` converts those into host wall-clock time and joins
them with the same run's telemetry — producing a per-core, per-kernel **wait vs. active vs.
overhead** breakdown, plus new charts in `FiguresNew/`.

## 2. Pipeline overview

```
auto.py (orchestrator)
  ├── starts telemetry exe (background)
  ├── deletes any stale generated/profiler/.logs/profile_log_device.csv
  ├── runs high_power_matmul (foreground) -- app emits device profiler markers if
  │   TT_METAL_DEVICE_PROFILER=1 is set
  ├── waits, stops telemetry
  └── invokes parser.py, passing --device-profiler-csv if the file exists

parser.py (analysis engine)
  ├── existing: telemetry.txt + summary.txt -> program_intervals.csv + Figures/*.png
  └── new: + profile_log_device.csv -> kernel_intervals.csv, kernel_utilization.csv,
           FiguresNew/*.png
```

Nothing about the *existing* telemetry-only analysis changed — if you don't pass
`--device-profiler-csv` (or the app didn't produce one), `parser.py` behaves exactly as before.

## 3. Device-side markers (what the kernels emit)

Each kernel marks its own start/end plus an aggregate wait-vs-active cycle count, once per
invocation (not per-iteration — see §7 for why):

| Kernel  | Markers | "Wait" = blocked in... | "Active" = time in... |
|---------|---------|-------------------------|------------------------|
| Reader  | `READER_KERNEL_START/END`, `READER_WAIT_CYCLES`, `READER_TRANSFER_CYCLES` | `cb_reserve_back` (input CB full) | `noc_async_read_tile` + barrier |
| Compute | `COMPUTE_KERNEL_START/END`, `COMPUTE_WAIT_CYCLES`, `COMPUTE_COMPUTE_CYCLES` (once **per TRISC** — 3x per core) | `cb_wait_front` (no input yet) | `matmul_tiles` + `pack_tile` |
| Writer  | `WRITER_KERNEL_START/END`, `WRITER_WAIT_CYCLES`, `WRITER_TRANSFER_CYCLES` | `cb_wait_front` (no output tile yet) | `noc_async_write_tile` + barrier |

Source: `tt_metal/programming_examples/high_power_matmul/kernels/{dataflow,compute}/*.cpp`.
The host (`high_power_matmul.cpp`) drains these via `ReadMeshDeviceProfilerResults(*mesh_device)`
after every grid run, into `generated/profiler/.logs/profile_log_device.csv`.

**Compute runs on 3 separate physical cores** (TRISC0=unpack, TRISC1=math, TRISC2=pack) — the
*same* C++ source is compiled 3 times, and tt-metal's own profiler tags each marker with which
one recorded it (the `RISC processor type` column). Only TRISC1 calls `matmul_tiles`; the three
roles have very different wait/active profiles and must **not** be averaged together (see §6).

**Required env vars to actually capture these markers:**
```bash
export TT_METAL_DEVICE_PROFILER=1   # turns on device profiling at all
export TT_METAL_PROFILER_SYNC=1     # converts device cycles to host wall-clock time
```
Without these, the app still runs fine, it just won't produce `profile_log_device.csv` (or it
will exist but be unusable for time conversion) — `auto.py` detects this and skips the
kernel-level analysis with a warning instead of failing.

## 4. New `parser.py` functionality

### 4.1 New CLI arguments

| Argument | Meaning |
|---|---|
| `--device-profiler-csv PATH` | Path to `profile_log_device.csv`. When given together with `--program-log`, enables everything in this document. |
| `--kernel-csv PATH` | Optional override for the *old* collapsed per-kernel CSV output path (default `<output>/<subdir>/kernel_intervals.csv`). |

Everything else (`-i`, `-o`, `--subdir`, `--program-log`, `--slot-ms`, `--trim-ms`,
`--device-id`, `--dpi`, `--program-csv`) is unchanged from before.

### 4.2 New outputs

Written to `<output-root>/<subdir>/` alongside the existing `program_intervals.csv`/`Figures/`:

- **`profile_log_device.csv`** — a copy of the input device-profiler CSV (for reproducibility, same pattern as the existing telemetry/summary copies).
- **`kernel_intervals.csv`** — one row per `(grid, kernel)`, collapsed across all cores of that kernel type: `grid, cores, kernel, start_time, end_time, duration_s`. This is the coarse, whole-kernel view (kept for backward compatibility with the first version of this feature).
- **`kernel_utilization.csv`** — the real new data: one row per `(grid, kernel, core_x, core_y, risc)`:

  | Column | Meaning |
  |---|---|
  | `kernel_start`, `kernel_end` | That core's own start/end, converted to host wall-clock time |
  | `lifetime_s` | `kernel_end - kernel_start` |
  | `wait_s` / `wait_pct` | From `*_WAIT_CYCLES`, converted to seconds / % of `lifetime_s` |
  | `active_s` / `active_pct` | From `*_TRANSFER_CYCLES` or `*_COMPUTE_CYCLES` |
  | `overhead_s` / `overhead_pct` | **Derived**, not measured: `max(0, lifetime_s - wait_s - active_s)` — see §7 |

- **`FiguresNew/`**:
  - `reader_utilization.png`, `writer_utilization.png` — stacked bar (active/overhead/wait %) vs. grid size, averaged over cores.
  - `compute_unpack_utilization.png`, `compute_math_utilization.png`, `compute_pack_utilization.png` — same, but **one file per TRISC role** (never blended — see §6).
  - `combined_utilization_pct.png` / `combined_utilization_seconds.png` — Reader / Compute(TRISC1-math) / Writer **side by side** on one chart, active+overhead+wait stacked. The seconds version adds a 4th "Run duration" reference bar per grid (§8).

### 4.3 Key functions (if you need to modify this further)

| Function | Purpose |
|---|---|
| `load_device_kernel_markers()` | Parses `profile_log_device.csv`. Note: `skipinitialspace=True` is required — the CSV header uses `", "` separators, which otherwise makes every column key have a leading space. |
| `compute_run_windows()` / `find_run_index()` | Establishes a per-run `(anchor_cycle, freq_hz)` fit from the *global* min(`*_KERNEL_START`)/max(`*_KERNEL_END`) across all 3 kernels for that run, then assigns any marker to a run by checking which run's cycle window contains it (**not** by core-coordinate range — `core_x`/`core_y` in the CSV are physical NOC coordinates, not the logical grid coordinates used to launch kernels). |
| `compute_kernel_intervals()` | Old collapsed-per-kernel view → `kernel_intervals.csv`. |
| `compute_kernel_utilization_rows()` | New per-core view → `kernel_utilization.csv`, including the derived-overhead calculation and the sanity clamp described in §7. |
| `aggregate_kernel_utilization()` / `plot_kernel_utilization()` | Per-`(grid, kernel, risc)` averages and the 5 individual charts. |
| `aggregate_combined_utilization()` / `plot_combined_utilization()` | The 2 combined charts, including the "Run duration" reference bar logic. |

## 5. `auto.py` changes

- New `--device-profiler-csv PATH` argument (default: `<app-workdir>/generated/profiler/.logs/profile_log_device.csv`).
- **Deletes** that file (if present) immediately before starting the app, so it only ever contains this run's fresh data (tt-metal itself also resets it on first `DeviceProfiler` construction, but this is a second layer of safety against stale data from a previous manual run).
- After the app finishes, checks whether the file now exists; if so, passes it to `parser.py` via `--device-profiler-csv`. If not (e.g. `TT_METAL_DEVICE_PROFILER` wasn't set), logs a note and skips it — the rest of the pipeline still runs normally.

No other changes — the existing `run_all.py` / `compare_runs.py` are untouched and unaffected.

## 6. Why compute has 3 separate charts, and why they can't be summed

`mm_power.cpp` is compiled once but **runs independently on 3 physical cores**
(TRISC0/1/2). Comparing/summing their percentages across roles is like adding two different
people's hours in the same meeting — each has its own 100% (its own lifetime), and they
overlap in wall-clock time rather than partitioning it. Empirically:

| Role | Typical wait% | Typical active% | What it's actually doing |
|---|---|---|---|
| TRISC0 (unpack) | ~90%+ | ~2% | Feeds tiles from L1 into the FPU's source registers |
| TRISC1 (math) | ~0.2% | 57–85% | The only role that calls `matmul_tiles` |
| TRISC2 (pack) | ~0.2% | ~0.3% | Packs FPU results back into the output CB |

Same logic applies across kernels: reader/compute/writer run **concurrently** on separate
cores for virtually the entire grid run (tightly coupled via the depth-2 circular buffers), so
their wait/active percentages are each "% of *that kernel's own* lifetime" — not one shared
100% budget.

## 7. Why "overhead" is *derived*, not measured — and what it really means

An earlier version measured a third on-device "overhead" bucket directly (wrapping
`tile_regs_acquire/commit/wait/release`, `cb_pop_front`, index arithmetic, etc. in their own
`read_wall_clock_cycles()` pairs). It was unreliable and read far too small even after adding
explicit compiler barriers (`asm volatile("" ::: "memory")`), because those spans have **no
real hardware operation** between the two timestamp reads — nothing forces the compiler to
keep surrounding (non-volatile) arithmetic from drifting relative to them, and cycle-counter
granularity compounds the error. `wait`/`active` were never affected by this, because they're
each bounded by a real hardware call (`cb_reserve_back`/`cb_wait_front` on one side,
`noc_async_read/write`/`matmul_tiles`/`pack_tile` on the other), which acts as a genuine
barrier.

**Fix: derive it instead.** `overhead_s = max(0, lifetime_s - wait_s - active_s)`. This is
exact by construction, since `lifetime_s` (from `KERNEL_START`/`KERNEL_END`) and `wait_s`/
`active_s` are all independently measured across real hardware operations. All on-device
overhead-measurement code and the `HIGH_POWER_MEASURE_OVERHEAD` flag that gated it have since
been **removed** — don't look for them, they no longer exist.

One consequence worth knowing: for TRISC1/TRISC2, "overhead" often dominates once
`HIGH_POWER_DISABLE_COMPUTE=1` removes their real work (§9) — this does *not* mean they're
doing pointless work. It means their *real* idle/handshake time isn't captured by the `wait`
label (which only brackets `cb_wait_front`, and that call resolves almost instantly on
TRISC1/TRISC2 — the real blocking likely happens inside `tile_regs_acquire`/`tile_regs_wait`,
which aren't separately instrumented). "Overhead" is a catch-all for "everything not named
wait or active," not a claim about *why* that time is spent.

Two other bugs fixed along the way, in case you see old data or old commits referencing them:
- **Unsigned 64-bit underflow**: a rare torn 32-bit-low/32-bit-high register read could produce
  `later < earlier` in a cycle delta, and `later - earlier` on `uint64_t` wrapped to ~`2^64`.
  Fixed with `safe_delta(later, earlier)` (clamps to 0) in all three kernels.
- **CSV `skipinitialspace`**: the device CSV header has `", "` separators; without
  `skipinitialspace=True`, every `csv.DictReader` key except the first had a leading space,
  silently matching nothing.

## 8. The "Run duration" reference bar

`combined_utilization_seconds.png` adds a 4th bar per grid: the max `lifetime_s` across
Reader/Compute(TRISC1)/Writer for that grid (they're all expected to be equal, since the three
kernels start/finish together). Since reader/compute/writer run *concurrently*, none of their
own active+overhead+wait stacks should exceed this reference — if one does, something's wrong
(mismatched run assignment, a fresh measurement bug, etc.).

## 9. Power-experiment flags on `high_power_matmul` itself

These are separate from the profiler markers above — they change what the kernels *do*, to
isolate where power is actually going. All are env vars read by `high_power_matmul.cpp` at
startup; none require a rebuild of the host binary (kernels are JIT-compiled, so a plain rerun
picks them up — a host rebuild is only needed if `high_power_matmul.cpp` itself changed).

### 9.1 Individual flags

| Env var | Effect |
|---|---|
| `HIGH_POWER_DISABLE_READER=1` | Reader still cycles `cb_reserve_back`/`cb_push_back` on both input CBs (so compute never deadlocks) but skips the real `noc_async_read_tile` + barrier. Input tiles contain stale L1 data. |
| `HIGH_POWER_DISABLE_COMPUTE=1` | Compute still cycles `cb_wait_front`/`cb_pop_front`, `tile_regs_acquire/commit/wait/release`, `cb_reserve_back`/`cb_push_back` (so reader/writer never deadlock) but skips `matmul_tiles` + `pack_tile`. Output tiles contain garbage. |
| `HIGH_POWER_DISABLE_WRITER=1` | Writer still cycles `cb_wait_front`/`cb_pop_front` (so compute never deadlocks) but skips `noc_async_write_tile` + barrier. Output DRAM buffer stays stale/uninitialized. |
| `HIGH_POWER_WRITE_AMPLIFICATION_PCT=<pct>` | For any real K, reader does `2*Kt` NoC reads per output tile but writer only ever did 1 — this re-writes each output tile `round((pct/100) * 2*Kt)` times (min 1) to load up the write-side NoC path symmetrically. `100` = match the reader's own NoC read volume exactly. Purely a power-stress knob; re-writes the *same* tile to the *same* address, so it doesn't affect correctness. Unset/0 = normal (1 write/tile). |

None of these need a correctness check on the result — `high_power_matmul.cpp` never verifies
`result_vec` against expected values, so stale/garbage output data is harmless; don't use these
modes for anything other than a power comparison. All four can be combined for finer manual
control; `POWER_CASE` (below) overrides all of them at once with one of 5 pre-defined
combinations.

### 9.2 `POWER_CASE` — single-flag control for the 5 canonical scenarios

Setting `POWER_CASE` **overrides all four flags above** for one of 5 pre-defined scenarios:

| `POWER_CASE` | Reader | Compute | Writer | Write amplification | Canonical subdir (used by `run_power_cases.py`) |
|---|---|---|---|---|---|
| `0` | real | real | real | off (baseline — everything as designed) | `regular` |
| `1` | real | real | real | 100% (writer stressed to match reader's NoC volume) | `writer_amp` |
| `2` | real | **idle** | real | 100% | `compute_idle` |
| `3` | **idle** | **idle** | real | 100% | `reader_compute_idle` |
| `4` | **idle** | real | real | 100% | `reader_idle2` |

"idle" means that kernel still runs its normal circular-buffer handshake (so the other kernels
never deadlock) but skips its real NoC read/write or FPU work.

```bash
POWER_CASE=2 ./build/programming_examples/metal_example_high_power_matmul 4096 8192 8192 160
```

Cases `0`–`3` each remove one more "real" component while keeping writer maximally stressed,
letting you isolate: full baseline (0) → does adding max write traffic on top change anything
(1) → is it just reader+writer NoC traffic with no real FPU work (2) → is it *only* writer NoC
traffic, everything else idle (3).

`POWER_CASE=4` isolates the reader's own NoC read cost while keeping compute and writer fully
stressed: reader still does `cb_reserve_back`/`cb_push_back` on both input CBs, so compute still
gets a tile "delivered" every iteration and runs its full `matmul_tiles`/`pack_tile` work on
whatever stale data is already sitting in that L1 buffer — compute is stimulated exactly as if
reader had sent it real data, just without any actual DRAM traffic on the read side. Combined
with `2`/`3`, this gives four single-variable comparisons against the `1` baseline: turn off
reader only (`4`), compute only (`2`), or both (`3`), all at the same
100%-write-amplification writer load.

If `POWER_CASE` is unset, the four individual flags in §9.1 are used as-is (all default to
"real"/off) for finer manual control.

Prints a confirmation line on startup, e.g.:
```
POWER_CASE=2 -- reader=real compute=idle writer=real write_amplification_pct=100
```

### 9.3 Sweeping the scenarios automatically

Comparing cases requires giving each one a clean electrical/thermal baseline (`tt-smi -r`)
before it runs, then pointing `auto.py` at a `--subdir` that identifies the case, once per
`POWER_CASE` value. [`run_power_cases.py`](../run_power_cases.py) automates that whole sequence
— reset, export, `auto.py` per case, cases-file generation, `compare_runs2.py` — in one
command. See the `run_power_cases.py` section of the [README](../README.md#run_power_casespy--power_case-sweep--comparison).

## 10. Full example: one complete run with everything on

```bash
# Activate the Python environment that has tt-metal's Python deps (tt_umd, etc.)
source /path/to/tt-metal-venv/bin/activate

# Point this at your tt-metal checkout
export TT_METAL_HOME=/path/to/tt-metal

export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_SYNC=1
export POWER_CASE=1        # or any of the individual HIGH_POWER_* flags instead

python3 auto.py \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --telemetry-freq 50 \
  --app-exe "$TT_METAL_HOME"/build_Release/programming_examples/metal_example_high_power_matmul \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/tt-metal-venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out \
  --subdir power_case_1_test \
  --slot-ms 1 \
  --device-id 0 \
  --trim-ms 1.0 \
  --app-args 4096 8192 8192 160
```

`--app-args` must be last (it consumes everything after it). Results land in
`out/power_case_1_test/`: `program_intervals.csv` + `Figures/` (existing telemetry analysis),
`kernel_intervals.csv` + `kernel_utilization.csv` + `FiguresNew/` (new device-side analysis).

### Re-running just the parser (no hardware needed)

If you already have `telemetry.txt` + `summary.txt` + `profile_log_device.csv` from a previous
run (e.g. you only changed `parser.py`, not the kernels/app), you don't need to re-run the
hardware — just re-invoke the parser directly against the existing files, writing back into
the same run directory:

```bash
python3 parser.py \
  -i out/power_case_1_test/telemetry.txt \
  --program-log out/power_case_1_test/summary.txt \
  -o out \
  --subdir power_case_1_test \
  --slot-ms 1 --device-id 0 --trim-ms 1.0 \
  --device-profiler-csv out/power_case_1_test/profile_log_device.csv
```
