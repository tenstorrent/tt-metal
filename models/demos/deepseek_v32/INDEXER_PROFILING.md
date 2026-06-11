# indexer_score — device-zone profiling & running tests

Short how-to for timing a region of a compute/dataflow kernel with a **scoped
device zone** and running the `indexer_score` tests (incl. under Tracy).

## Scoped device zones

`DeviceZoneScopedN("NAME")` is an RAII timer: it stamps a start cycle where it is
declared and an end cycle at scope exit. It records **only while the per-RISC L1
marker buffer has room**, so in a hot loop it captures the first N iterations and
then silently stops — that is the difference from the *guaranteed* whole-kernel
zones (`DeviceZoneScopedMainN` / `...MainChildN`, which use reserved slots). For a
loop, give the region its own block so the zone covers exactly that region:

```cpp
#include "tools/profiler/kernel_profiler.hpp"   // in the kernel .cpp

// ... inside the per-tile loop:
{
    DeviceZoneScopedN("PACK_UNTILIZE");
    compute_kernel_lib::untilize<1, cb_acc, cb_out>(1);
}   // end cycle stamped here
```

Notes:
- The zone fires on whichever RISC executes that source region (e.g. the untilize
  region shows up under `TRISC_0`).
- The measured time is the scope **wall-time**, so it includes any CB back-pressure
  stalls inside the region (e.g. waiting on `cb_out`). To isolate pure compute,
  measure with data movement disabled.
- Zones only emit when the run is profiled (see Tracy below) and the kernels are
  JIT-rebuilt with the profiler enabled (a profiled run forces this).

## Running the tests

Always go through the device-aware wrapper (flock + auto-reset; never `tt-smi -r`):

```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh [--dev] [--run-all] <test_file>::<test>[<id>]
```

Useful `indexer_score` selectors (`tests/nightly/blackhole/sdpa/test_indexer_score.py`):

```bash
# accuracy (PCC + exact -inf map), 16-head TP shard and 64-head whole indexer
scripts/run_safe_pytest.sh --run-all tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_production
# wall-clock latency (in-process; env knobs apply)
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_indexer_score.py::"test_indexer_score_production_perf[rank7-k_bfp8-heads16]"
# sp7 matmul math-utilization (spawns the inner perf test under the device profiler)
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_indexer_score.py::"test_indexer_score_sp7_math_util[heads16_k_bfp8]"
```

Kernel edits do **not** need `build_metal.sh` — kernels JIT-rebuild per run.

## Running under Tracy (captures device zones → .tracy + .csv)

`run_safe_pytest.sh` uses `flock /tmp/tt-device.lock`; hold the same lock so a
direct Tracy run stays cooperative with other agents:

```bash
flock /tmp/tt-device.lock bash -c '
  source python_env/bin/activate
  python -m tracy -r -p -o generated/profiler/indexer_zone \
    -m "pytest tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_sp7_perf_impl[heads16_k_bfp8]"
'
```

`-p` profiles only enabled zones and forces a profiler-enabled JIT rebuild; `-r`
generates the ops report. Artifacts land under the `-o` folder:

| file | what |
|---|---|
| `.logs/tracy_profile_log_host.tracy` | Tracy GUI capture |
| `.logs/profile_log_device.csv` | per-zone start/end markers (the raw zone data) |
| `.logs/cpp_device_perf_report.csv` | op-level device durations (per-RISC) |
| `reports/<ts>/ops_perf_results_*.csv` | post-processed op perf CSV |

## Reading a zone's duration from the device CSV

`profile_log_device.csv` columns: `…, time[cycles] (6), …, zone name (11), type (12), source line (13), …`.
Average a zone (BH clock = 1350 MHz):

```bash
awk -F, '$11=="PACK_UNTILIZE"{
    if($12=="ZONE_START"){t=$6}
    else if($12=="ZONE_END"){d=$6-t; sum+=d; n++}
} END{ printf "pairs=%d  avg=%.1f cyc (%.1f ns)\n", n, sum/n, (sum/n)/1.35 }' \
  generated/profiler/indexer_zone/.logs/profile_log_device.csv
```
