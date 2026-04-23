# Profile tt-metal ops and kernels

## Prerequisites

```bash
pip install tt-perf-report
```

Install once in the workspace's venv. `tt-perf-report` is the CLI that renders
device-timing CSVs as a ranked report.

## Run a test with profiling

```bash
python -m tracy -p -r -v -m pytest <test_path> -k <filter> -v
```

Flags: `-p` enable profiling, `-r` generate CSV report, `-v` verbose. Internally
sets `TT_METAL_DEVICE_PROFILER=1` and runs the wrapped command. The wrapped
command is a full pytest (or other) invocation — quote it if it contains spaces.

### Constraints (enforced by the device)

- Device profiler, `TT_METAL_DPRINT_CORES`, and `TT_METAL_WATCHER` conflict —
  do not enable more than one at a time. Unset `TTNN_CONFIG_PATH` if it was
  set for memory/visualizer runs.
- First run of a test populates the program cache — host times are inflated.
  The test should iterate at least twice; analyze only the second iteration.
- Tracy buffers up to 1000 ops per device before dropping. For longer tests,
  call `ReadDeviceProfilerResults(device)` periodically from the test.
- Grayskull requires a full host reboot after `tt_smi` reset to realign tensix
  timer starts. Wormhole/Blackhole do not.

## Output layout

Each run creates a timestamped folder:

```
$TT_METAL_HOME/generated/profiler/reports/<timestamp>/
  ops_perf_results_<timestamp>.csv        # per-op device + host timing (the input for tt-perf-report)
  profile_log_device.csv                  # raw device-zone data
  <name>.tracy                            # Tracy GUI timeline
```

`<timestamp>` is the most recent subdirectory under `generated/profiler/reports/`
— list by mtime to find it after a run.

## Analyze with tt-perf-report

```bash
tt-perf-report $TT_METAL_HOME/generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

Produces a ranked table: op code, device FW duration, per-RISC durations, core
count, math fidelity, parallelization strategy.

### Escape hatch

If `tt-perf-report` crashes on a custom op (tends to happen on unfamiliar op
codes):

```bash
tt-perf-report --no-stacked <csv>
```

### Fallback: read the CSV directly with pandas

`tt-perf-report` 1.2.3 crashes with `Unknown math fidelity` on any row whose
`MATH FIDELITY` column is `HiFi3`. `--no-stacked` does not help — the crash
is in the main analysis loop, not the stacked-report section. Also happens
on other exotic fidelity or op-code combinations.

When the report bails, read the CSV directly — the columns the report uses
are all present in the raw file:

```python
import pandas as pd
csv = "$TT_METAL_HOME/generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv"
df = pd.read_csv(csv)

# Top ops by device FW duration
top = (df.groupby("OP CODE")["DEVICE FW DURATION [ns]"]
         .agg(["sum", "count", "mean"])
         .sort_values("sum", ascending=False)
         .head(10))

# Per-matmul slice with everything needed for a bottleneck read
mm = df[df["OP CODE"].str.contains("Matmul", na=False)][[
    "OP CODE", "DEVICE FW DURATION [ns]", "CORE COUNT",
    "MATH FIDELITY", "PARALLELIZATION STRATEGY",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
]]
```

This is sufficient to produce the same profile note as the report-driven
path (top ops, per-RISC split, fidelity/cores/strategy). It does **not**
compute the `DRAM %` / `FLOPs %` utilization columns — those require the
op's theoretical work volume, which the report knows per op code. For a
matmul, compute it manually from `M × K × N × 2` FLOPs and the `DEVICE FW
DURATION [ns]`.

## Device zones from kernel code

To profile a specific kernel region, add a scoped zone to the kernel source:

```cpp
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    DeviceZoneScopedN("MY_ZONE");
    // code to profile
}
```

Zones appear in the Tracy GUI timeline and in per-RISC CSV columns. Adds
measurable overhead — use only on hot paths being investigated.

## See also

- Device program profiler docs: `docs/source/tt-metalium/tools/device_program_profiler.rst`
- TTNN op profiling docs: `docs/source/ttnn/ttnn/profiling_ttnn_operations.rst`
- Onboarding exercise: `tt-shield/ttnn-onboarding/e06_profiling/README.md`
- Programming examples: `tt_metal/programming_examples/profiler/`
- tt-perf-report source: https://github.com/tenstorrent/tt-perf-report
