# `op_to_op_latency` — CI microbenchmark

Tracks **op-to-op latency** (the on-chip gap between one program finishing and the
next program starting) over time in CI, as part of extending the runtime
microbenchmark suite ([issue #46305](https://github.com/tenstorrent/tt-metal/issues/46305)).

This is the CI/regression packaging of the op-to-op work. It runs **one pinned
steady-state config**, captures profiler data, post-processes it in Python, and
gates a single metric against a per-arch golden.

The buffer-sizing auto-search (`--buffer-tune`) and the broader gap-decomposition
study live in the research benchmark PR
([#44048](https://github.com/tenstorrent/tt-metal/pull/44048)) and are intentionally
not part of this CI binary — an auto-search doesn't yield a stable number to gate on.
The binary does keep a few *fixed-config* knobs (NoC assignment, active-core count,
`--read-only`, CB depths, reader batch/double-buffer modes) that are unused by the
default config above but can seed future pinned CI entries (e.g. a DRAM read-BW or
NoC-direction metric with its own golden).

## Files

```
test_op_to_op_latency.cpp     host benchmark (reader -> CB -> compute -> CB -> writer, back-to-back N programs)
kernels/
├── reader_interleaved.cpp    NCRISC reader
├── writer_interleaved.cpp    BRISC writer
└── compute_copy_with_nops.cpp  TRISC copy + tunable NOP spin
op_to_op_postprocess.py       turns the profiler CSVs into scalar CI metrics; with --golden it gates vs golden
op_to_op_golden.json          Wormhole golden (values populated from a real CI run)
op_to_op_blackhole_golden.json  Blackhole golden
```

## What is measured

The test binary dumps two raw profiler CSVs after the run:

| CSV | Source |
|-----|--------|
| `profile_log_device.csv` | device profiler (standard `{BRISC,NCRISC,TRISC}-KERNEL` zones) |
| `profile_log_device_rt.csv` | realtime profiler (per-program go/done) |

`op_to_op_postprocess.py` reduces them to:

| Metric | Role | Meaning |
|--------|------|---------|
| `official_op2op_us` | **GATED** | Per-core adjacent-op gap from standard KERNEL zones: last KERNEL end(k) → first DM-KERNEL start(k+1). Canonical tools/tracy device number; works on every platform; no custom markers. |
| `rt_gap_to_next_go_ns` | **TRACKED** | Chip-dispatcher done→go gap from the realtime profiler. Cleaner/absolute, but RT is only active on some setups (not T3K/remote/ETH dispatch), so it is recorded where available and **not gated**. |
| `device_kernel_dur_us` | context | Per-op kernel span (first KERNEL start → last KERNEL end). |
| `pack_to_unpack_op2op_us` | context | Per-core pack-finish → next-unpack-start (the research benchmark's op2op definition). |

We gate on the KERNEL-zone metric because it is portable across all CI
single-card platforms; the RT metric is tracked alongside it where the realtime
profiler is active.

## Where it runs

On the **(Runtime) Performance Tests** pipeline (`.github/workflows/runtime-perf-tests.yaml`,
job `runtime-perf-profiler-tests`), via `tests/pipeline_reorg/runtime_perf_profiler_tests.yaml`
(entry `runtime_perf_op_to_op_latency`). That job uses a dedicated profiler build
(`build-artifact-profiler`, `tracy: true` / `ENABLE_TRACY=ON`) so the non-profiler
dispatch/bandwidth microbenchmarks in `runtime_perf_tests.yaml` are not perturbed.
SKUs: `wh_n300_civ2`, `bh_p150_perf`. Budget subheading: `perf_profiler`.

## Run locally

Needs a Tracy-enabled build (default; i.e. do **not** pass `build_metal.sh --disable-profiler`):

```bash
export TT_METAL_HOME=$(pwd)
cmake --build build --target test_op_to_op_latency -j

# the exact CI flow: run the pinned config, then post-process + gate vs golden
TT_METAL_DEVICE_PROFILER=1 ./build/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --use-trace --trace-warmup-replays 2 --num-programs 8 --num-pages-per-core 4 \
  --compute-nops 2000 --use-device-profiler --use-realtime-profiler

# print metrics only (no gate)
python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/op_to_op_postprocess.py --min-prog-id 3

# print metrics and gate against the golden (record mode passes while golden is null)
python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/op_to_op_postprocess.py \
  --min-prog-id 3 --golden tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/op_to_op_golden.json
```

## Populating / updating the golden

The golden files ship with `null` values ("record mode"): until a golden is
populated the post-proc **only prints the measured metrics and skips the gate**
(passes), so the test can land before we have silicon numbers.

To enable the gate:
1. Run the binary + post-proc on the target SKU.
2. Copy the measured `official_op2op_us` into the matching golden file's
   `golden.official_op2op_us` (WH → `op_to_op_golden.json`, BH →
   `op_to_op_blackhole_golden.json`), and optionally record the tracked values.
3. Tune `golden.tolerance_pct` (start loose, tighten once run-to-run spread is known).

## Notes

- The pinned config is intentionally fixed for run-to-run comparability. Do not
  add sweeps here — sweeping/tuning belongs in the research benchmark.
- `--compute-nops` is set so the program is comfortably long vs dispatch noise;
  the exact value is arch-dependent and folded into the golden, so it can be
  retuned when populating the golden.
- The realtime profiler is guarded by `IsProgramRealtimeProfilerActive()` in the
  binary; on platforms where it is inactive the RT metric is simply empty and the
  KERNEL-zone gate still applies.
