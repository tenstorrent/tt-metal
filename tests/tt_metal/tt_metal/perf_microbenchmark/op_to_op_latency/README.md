# `op_to_op_latency` — CI microbenchmark

Tracks **op-to-op latency** (the on-chip gap between one program finishing and the
next program starting) over time in CI, as part of extending the runtime
microbenchmark suite ([issue #46305](https://github.com/tenstorrent/tt-metal/issues/46305)).

This is the CI/regression packaging of the op-to-op work. It runs **one pinned
steady-state config**, captures profiler data, post-processes it in Python, and
gates a single metric against a per-arch golden. (The broader op-to-op
optimization study — buffer tuning, NoC sweeps, gap decomposition — lives in the
research benchmark PR and is intentionally not part of this CI test.)

## Files

```
test_op_to_op_latency.cpp     host benchmark (reader -> CB -> compute -> CB -> writer, back-to-back N programs)
kernels/
├── reader_interleaved.cpp    NCRISC reader
├── writer_interleaved.cpp    BRISC writer
└── compute_copy_with_nops.cpp  TRISC copy + tunable NOP spin
op_to_op_postprocess.py       turns the profiler CSVs into scalar CI metrics
test_op_to_op_ci.py           pytest: run pinned config -> post-process -> gate vs golden
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

On the **single-card profiler pipeline** (`tests/pipeline_reorg/single_card_profiler_tests.yaml`,
job "Op-to-op latency"), which builds with `ENABLE_TRACY=ON` and already runs the
device + realtime profiler suites. SKUs: `wh_n150_civ2`, `wh_n300_civ2`,
`bh_p100a_civ2_viommu`, `bh_p150b_civ2_viommu`.

## Run locally

Needs a Tracy-enabled build (default; i.e. do **not** pass `build_metal.sh --disable-profiler`):

```bash
export TT_METAL_HOME=$(pwd)
cmake --build build --target test_op_to_op_latency -j

# via the CI wrapper (runs the pinned config, post-processes, gates vs golden)
pytest tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_ci.py

# or drive the binary + post-proc directly
TT_METAL_DEVICE_PROFILER=1 ./build/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --use-trace --trace-warmup-replays 2 --num-programs 8 --num-pages-per-core 4 \
  --compute-nops 2000 --use-device-profiler --use-realtime-profiler

python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/op_to_op_postprocess.py --min-prog-id 3
```

## Populating / updating the golden

The golden files ship with `null` values ("record mode"): until a golden is
populated the pytest **only prints the measured metrics and skips the gate**, so
the test can land before we have silicon numbers.

To enable the gate:
1. Run the pytest (or the binary + post-proc) on the target SKU.
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
