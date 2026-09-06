# Capturing a profile

## Capture

```bash
timeout 1800 ./python_env/bin/python -m tracy -p -r -v \
  -m pytest <test_path> -k <filter> -s --timeout 900 &> tracy.log
```

| Flag | |
|---|---|
| `-p` `-r` `-v` `-m` | Profile enabled zones · generate ops report · verbose · profile a module. Sets `TT_METAL_DEVICE_PROFILER=1` internally |
| `-o` `-n` | Output folder, name suffix |
| `--no-device` | Host-only |
| `--op-support-count` | Raise the op ceiling |
| `--profile-dispatch-cores` | Include dispatch cores |

Always timeout-gated (`../shared/device-hangs.md`). Redirect to a file —
**never pipe to `tail -N`**; buffering leaves the log empty until exit.

**Profiler, watcher and DPRINT conflict** — all three consume device SRAM. Unset
`TTNN_CONFIG_PATH` too. **All device durations zero** means one is still set.

## Signposts — how you get a warm window

```python
from tracy import signpost

model = build(...)                    # weight upload — outside
x = ttnn.from_torch(...)              # activation prep (tilize) — outside
_ = model(x); ttnn.synchronize_device(mesh)   # warm-up iteration — outside

signpost("start")                     # model code only, from here
_ = model(x)
ttnn.synchronize_device(mesh)
signpost("stop")
ttnn.ReadDeviceProfiler(mesh)
```

**Everything before `signpost("start")` must be construction, not inference.**
Weight upload and activation prep emit a run of `TilizeWithValPadding` /
`Untilize` / layout ops with large gaps between them; leave them inside the
window and they will dominate the aggregate and make data movement look like the
bottleneck.

```bash
tt-perf-report --print-signposts <csv>                          # what's in the file
tt-perf-report --start-signpost start --end-signpost stop <csv>
tt-perf-report --ignore-signposts <csv>                         # whole file, incl. weight upload
```

Without signposts the report prints *"No signposts found in the file. Using the
entire file for analysis"* and its aggregates include construction and weight
upload — which produced a wrong conclusion in this tree
(`reading-profiles.md`). Three lines in the test removes the problem class.
In-tree example: `tests/models/ltx/test_vae_ltx.py`.

## Device zones

C++ kernels use `DeviceZoneScopedN(zone_name)` — Tracy's `ZoneScopedN`
equivalent. The zone appears as a named row in
`generated/profiler/.logs/profile_log_device.csv` beside the default `BRISC-FW`
and `BRISC-KERNEL` markers, with source file and line. Only worth it when one
op's `DEVICE FW DURATION` is the bottleneck and you need to know which phase
inside it costs.

## The op buffer

Tracy buffers ~1000 ops per device before dropping. Both
`AssertionError: Device data missing: Op <id>` and "only a handful of ops
captured" are this.

**Shrink the scope. Do not reach for the dump.**

| Order | Action | Why |
|---|---|---|
| 1 | Profile a smaller scope — one layer, one down-block, one op | A smaller scope is a *better* profile: the ranking within one repeated unit is what generalizes |
| 2 | A single profiled forward after a warm-up | Device kernel durations are warm-independent once the program cache is populated. `test_vae_ltx.py` does this for the decoder |
| 3 | **Only if you genuinely cannot reduce the op count** — flush mid-run | `TT_METAL_PROFILER_MID_RUN_DUMP=1` + periodic `ttnn.ReadDeviceProfiler(mesh)` (C++: `tt::tt_metal::detail::ReadDeviceProfilerResults(device)`). `utils/sweep_mm_block_sizes.py` uses `PROFILER_DUMP_EVERY = 10` because a sweep genuinely needs many configs in one session. Adds host sync inside the measured region |

## Output

```
generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv   # input to tt-perf-report
generated/profiler/reports/<ts>/<name>.tracy                # GUI timeline
generated/profiler/.logs/profile_log_device.csv             # raw zone data

ls -dt $TT_METAL_HOME/generated/profiler/reports/*/ | head -1
```

## Analysis

```bash
tt-perf-report --start-signpost start --end-signpost stop <csv>
```

| Flag | Use |
|---|---|
| `--id-range 31-` / `-12` / `5-10` | Slice by op ID — the warm window when there are no signposts |
| `--group-by op\|memory\|category` | `category` splits compute / data movement / tensor |
| `--arch blackhole` | Auto-detected on new reports; drives the FLOPs%/DRAM% denominators |
| `--csv <file>` / `--summary-file <file>` | Machine-readable output for a sweep |
| `--tracing-mode` | Do not sort — for traced captures |
| `--no-advice` | Table only |
| `--no-summary` | Skip the summary — escape hatch when it crashes on an unfamiliar op code (`--no-stacked-report` is the deprecated spelling) |

**Fallback: read the CSV directly.** When the report cannot handle an op code at
all, every column it uses is in the raw file:

```python
import pandas as pd
df = pd.read_csv(csv)
warm = df.tail(300)          # prefer signposts or --id-range

top = (warm.groupby("OP CODE")["DEVICE FW DURATION [ns]"]
           .agg(["sum", "count", "mean"]).sort_values("sum", ascending=False).head(10))
top["pct"] = 100 * top["sum"] / warm["DEVICE FW DURATION [ns]"].sum()

gap = warm["OP TO OP LATENCY [ns]"]          # never skip this
print(f"gap median {gap.median()/1e3:.1f} us, mean {gap.mean()/1e3:.1f} us")
```

Gives the same per-op ranking, per-RISC split, fidelity and core counts. It does
**not** compute `DRAM %` / `FLOPs %` — those need the op's theoretical work
volume, which the report derives per op code from `--arch`.

## Measuring trace

Capture with the model's trace gate on (`LTX_TRACED=1` or equivalent), add
`--tracing-mode`, compare warm windows. Trace collapses op-to-op dispatch gaps;
it does not change device time per op. Needs `trace_region_size` in the device
fixture. Only after the gap distribution shows dispatch is a meaningful share
(`../tt-dit-performance/optimization-levers.md` § 7).
