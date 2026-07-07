---
name: core-occupancy
description: >
  Measure the core-occupancy / utilization efficiency of a TTNN op on a SPECIFIC input
  shape, empirically, via the Tracy device profiler. Use when you need to know how well an
  op fills the compute grid (are cores idle?) and balances work across the cores it does use,
  for a given shape — e.g. to detect grid under-utilization before proposing a
  parallelization/work-split refinement, or to validate one. Produces a single efficiency
  number that factors exactly into occupancy × balance, straight from the profiler CSVs with
  NO kernel edits. Occupancy is shape-dependent, so always measure per shape.
---

# core-occupancy

Empirically measure how well a TTNN op uses the compute grid **for one input shape**, from
the Tracy device profiler. No kernel edits — everything comes from the default per-RISC
`*-KERNEL` zones the profiler always emits.

## Unit of measurement: one test cell

Occupancy is a property of **this op × this shape/config × this grid** — not of the op in
general. So one measurement = **one test cell**:

- The agent (oriented by this skill) supplies **one command that runs the op exactly once** on
  the target shape/config — a single pytest node id, or a small "run this op once on shape X"
  script. For an op without such a driver, write a minimal one (invoke the op once;
  correctness check off).
- One run → one `{occupancy, balance, efficiency}` triple for that cell.
- Changing anything that alters work distribution — shape, tiling/chunk parameters, dtype,
  grid — is a **different cell**; re-measure. A *sweep* is just this skill looped over cells,
  one row each; the skill itself stays single-cell.

## The metric

Let `N` = available worker cores (the op's grid), `dᵢ` = per-core kernel duration (a core that
never launched has `dᵢ = 0`, so it is absent from the raw log but still counted in `N`):

```
efficiency  M = Σdᵢ / (N · max dᵢ) = mean_over_all_N(dᵢ) / max(dᵢ)
             = (n_busy / N)  ×  (mean_busy / max)
             =  occupancy    ×   balance
```

- Perfect (all `N` cores busy, equal duration) → **M = 1.0**
- Worst (1 core busy, rest idle) → **M = 1/N**
- 100% occupied but one core is a 2× straggler → **M ≈ 0.5** (catches imbalance, which raw
  `n_busy/N` misses)

**Always report all three** — `M`, `occupancy`, `balance` — because a low `M` alone cannot
distinguish *idle cores* (occupancy) from *uneven work* (balance), and they call for different
refinements:
- low **occupancy** → the op left cores idle → parallelize / split work across more cores.
- low **balance** → cores used, but unevenly → rebalance the work distribution.

## Prerequisites

- A **profiler-enabled build** (`./build_metal.sh --enable-profiler …`; check for
  `build/tools/profiler/bin/tracy-capture`).
- A way to **invoke the op exactly once on the target shape** — a pytest node id or a small
  script. This is the *only* op-specific input; everything else is universal.
- A device (the op's `device` fixture auto-opens the default one).

## Procedure

### 1. Run the op under Tracy

```
python3 -m tracy -p -r -o generated/profiler/<subdir> -a device_kernel_duration -t 5000 \
  -m "pytest '<node-id that runs the op once on the target shape>'"
```

- `-a device_kernel_duration` produces the per-op summary (incl. `CORE COUNT`).
- Run via **plain `pytest`/the tracy wrapper, NOT `run_safe_pytest.sh`** — the profiler spawns
  its own device subprocess; the flock wrapper would contend for the device.
- If the op has a driver test that already wraps `run_device_profiler` (e.g. a `*_perf` test),
  just run that test with plain `pytest -s`; it builds this command for you.

### 2. Locate the outputs

- **ops-perf CSV** (per-op summary): `generated/profiler/<subdir>/reports/<date>/ops_perf_results_<date>.csv`
  — columns include `OP CODE`, `CORE COUNT`, `AVAILABLE WORKER CORE COUNT`,
  `DEVICE KERNEL DURATION [ns]`, and `DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]`.
- **raw per-core zones**: `generated/profiler/<subdir>/.logs/profile_log_device.csv`
  — header line 1 is metadata (`ARCH`, `CHIP_FREQ[MHz]`, `Max Compute Cores`); line 2 is the
  real column header: `PCIe slot, core_x, core_y, RISC processor type, timer_id,
  time[cycles since reset], data, …, zone name, type, source line, source file, meta data`.

### 3. Read the denominator N

`N = AVAILABLE WORKER CORE COUNT` from the ops-perf CSV. **Do NOT use `Max Compute Cores`**
from the raw header — that is the physical maximum, while the op's grid may be smaller (a
sub-grid, or reduced because the op reserves some cores for other purposes).

### 4. Per-core durations from the raw CSV

Always derive per-core durations from the **raw** CSV — the ops-perf
`DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]` columns are frequently **NaN** (the per-core
analysis is silently ignored when `cpp_device_perf_report` is active, even with
`-a device_kernel_duration`), so do not rely on them.

For each core, kernel duration = its `*-KERNEL` zone span. Parse:

```python
import pandas as pd
f = "generated/profiler/<subdir>/.logs/profile_log_device.csv"
freq = float(open(f).readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])  # e.g. 1350
df = pd.read_csv(f, skiprows=1, skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
k = df[df["zone name"].astype(str).str.endswith("KERNEL")].copy()
k["t"] = k["time[cycles since reset]"].astype(float)
# duration per (core, RISC, zone) = end - start; per-core = span over its RISC kernels
g = k.groupby(["core_x", "core_y", "RISC processor type", "zone name"])["t"].agg(["min", "max"])
g["dur_ns"] = (g["max"] - g["min"]) / (freq / 1000.0)
percore = g.reset_index().groupby(["core_x", "core_y"])["dur_ns"].max()   # core's busy span
```

### 5. Compute and report

```python
N = ...            # AVAILABLE WORKER CORE COUNT, read from the ops-perf CSV
n_busy = percore.shape[0]
dmax   = percore.max()
occupancy  = n_busy / N
balance    = percore.mean() / dmax
efficiency = percore.sum() / (N * dmax)          # == occupancy * balance
print(f"occupancy={occupancy:.3f}  balance={balance:.3f}  efficiency={efficiency:.3f}")
```

Cross-check: `n_busy` (distinct cores in the raw log) should equal `CORE COUNT` in the
ops-perf CSV. If they disagree, trust the raw log and investigate.

## Gotchas / universal knowledge

- **Denominator = available grid, idle cores counted as 0.** Dividing by `N` (not `n_busy`) is
  exactly what makes `efficiency` penalize idle cores. Idle cores are simply absent from the
  raw CSV.
- **No source edit is required** for occupancy or balance — the default FW/`*-KERNEL` zones
  (≈20 marker rows/core) already carry per-core durations. Only reach for a temporary
  `DeviceZoneScopedN` marker if you need the *discrete count of work-units per core* (a
  separate, harder signal — mind the ~250-marker/core/launch budget and isolate it on its own
  compile-time gate). If you do add such a temporary marker, revert it via the shared
  snapshot-then-restore procedure in [`../shared/revert-temp-edits.md`](../shared/revert-temp-edits.md)
  — never `git checkout` a file that may hold the user's uncommitted work.
- **cycles → ns** via `CHIP_FREQ[MHz]` from the raw header (`ns = cycles / (MHz/1000)`).
- **per-core duration** = max over that core's RISC `*-KERNEL` zones (the core's wall-busy
  span). BRISC=writer, NCRISC=reader, TRISC_0/1/2=compute.
- A small (~1%) `core_x` gradient in per-core duration is normal (NoC distance to DRAM), not
  imbalance — don't over-read spreads near 1.0.
- **best-of-N**: device time drifts a few %; take the min of a few runs before trusting small
  differences.
- Occupancy is **shape-dependent** — this measures the shape you ran; re-measure per shape.

## Interpreting the result

Read the two factors, not just `efficiency` — they map to different levers:

| occupancy | balance | reading | lever |
|-----------|---------|---------|-------|
| ≈ 1 | ≈ 1 | grid full, work even | already well-utilized on this cell |
| ≈ 1 | < 1 | grid full but uneven work | **rebalance** — the work-units don't divide evenly across the cores (or a straggler dominates); adjust tiling/grid so units ≈ a multiple of the core count |
| < 1 | ≈ 1 | cores idle, the busy ones even | **parallelize** — the op left cores unused; split work across more cores |
| < 1 | < 1 | idle cores *and* uneven | both — usually parallelize first, then rebalance |

Also inspect the **per-core duration distribution**, not only the aggregate: a **bimodal**
split (two well-separated clusters) is the fingerprint of a work-count imbalance (some cores
got one more unit than others), whereas a single tight cluster with a small tail is just the
NoC gradient. The cluster ratio ≈ the per-core work-count ratio.
