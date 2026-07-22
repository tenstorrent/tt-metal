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

By default the kernels run **lean**: they emit only the profiler markers the CI metrics
actually consume (the firmware `{BRISC,NCRISC,TRISC}-KERNEL` zones for the gated metric,
plus compute `PROG_ID` / tile-0 first-math / pack-finish for `pack_to_unpack`). The two
lean compute markers are emitted with `DeviceRecordEvent` (event id only, no data payload)
rather than `DeviceTimestampedData`, so they write fewer words to the L1 profiler buffer and
perturb the op2op gap less; `op_to_op_postprocess.py` maps the event ids back to the
`TILE_IDX` / `FINISH_LAST_PUSH` names (`EVENT_NAMES`, kept in sync with the kernel `EV_*`
constants). The extra reader/writer `GO`/`DONE`/`BARRIER` markers and the per-tile compute
`TILE_IDX` + `MATH` zone used by the research BW/gap-decomposition analysis are gated behind
`--profile-detail` (off by default) so they don't add device cycles that perturb the gated
op2op number.

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

The CI run uses only the standard **device profiler**, so it behaves identically on
Wormhole and Blackhole. It dumps one raw CSV:

| CSV | Source |
|-----|--------|
| `profile_log_device.csv` | device profiler (standard `{BRISC,NCRISC,TRISC}-KERNEL` zones) |

The newer **realtime profiler** is intentionally *not* used by the CI command — it is
unsupported on some WH setups (and T3K/remote/ETH dispatch), so depending on it would make
the test platform-specific. It is still available for local/research use by adding
`--use-realtime-profiler` (which then also writes `profile_log_device_rt.csv`).

`op_to_op_postprocess.py` loads the device log through the official parser
(`tools/tracy/process_device_log.py`) rather than reading the CSV columns directly — that
module owns the on-disk schema, so the test stays robust to CSV layout changes — and
reduces the logs to:

| Metric | Role | Meaning |
|--------|------|---------|
| `official_op2op_us` | **GATED** | Per-core adjacent-op gap from standard KERNEL zones: last KERNEL end(k) → first DM-KERNEL start(k+1). Canonical tools/tracy device number; works on every platform; no custom markers. |
| `pack_to_unpack_op2op_us` | **GATED** | Per-core pack-finish → next-unpack-start (the research benchmark's op2op definition). Uses the lean compute markers, so it is portable across WH/BH. |
| `device_kernel_dur_us` | context | Per-op kernel span (first KERNEL start → last KERNEL end). |
| `rt_gap_to_next_go_ns` | optional | Chip-dispatcher done→go gap from the realtime profiler. Cleaner/absolute, but only produced when `--use-realtime-profiler` is passed (research); empty (n=0) in the CI run. |

Both gated metrics come from the standard device profiler, so they are portable across
all CI single-card platforms (Wormhole and Blackhole). The gate fails the job if *either*
metric drifts outside its golden band (`tolerance_pct`); the post-processor gates every
non-null value in the golden's `golden` block. A metric left `null` in the golden is in
record mode (printed, not gated), so the gate can be armed one metric at a time. The RT
metric is not part of the CI flow; it can be captured locally where the realtime profiler
is active.

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
# (add --use-realtime-profiler to also capture the optional RT metric where supported)
TT_METAL_DEVICE_PROFILER=1 ./build/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --use-trace --trace-warmup-replays 2 --num-programs 8 --num-pages-per-core 4 \
  --compute-nops 2000 --use-device-profiler

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
- The CI flow uses only the device profiler, so the gate is identical on Wormhole and
  Blackhole. The realtime profiler is optional (`--use-realtime-profiler`) and guarded by
  `IsProgramRealtimeProfilerActive()` in the binary — if requested where it is inactive it
  simply warns and skips, so the KERNEL-zone gate always applies.
