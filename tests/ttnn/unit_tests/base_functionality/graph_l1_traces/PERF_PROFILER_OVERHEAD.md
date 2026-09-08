# Eval performance-profiler overhead

Measured on 2026-09-08 on Blackhole, from commit `e488cbd956f4b61b8dd4101df87ccf0d31a654f5`.
This measures the existing eval "perf profiler" implemented by `eval/profiling.py`, independently
of the graph-capture L1 profiler.

## Result

The workload is the same 50-distinct-program RMSNorm matrix used by
`GRADED_RUN_L1_OVERHEAD.md`: FLOAT32 tiled `[1, 1, 32, W]`, with `W=32..1600` in steps of 32.
Every mode was run three times in its own graded process after precompile. The profiler-compatible
runs reported 269/269 persistent JIT-cache hits. The table reports the median aggregate pytest
call-hook time.

| Mode | 50-call time | Mean/case | Delta from profiler off |
|---|---:|---:|---:|
| Perf profiler off | 381.872 ms | 7.637 ms | -- |
| Device profiler enabled, no per-test reads | 865.855 ms | 17.317 ms | +9.680 ms / +126.7% |
| Complete eval perf profiler | 2,068.947 ms | 41.379 ms | **+33.741 ms / +441.8%** |
| Complete profiler, C++ INFO logging suppressed | 2,017.153 ms | 40.343 ms | +32.706 ms / +428.3% |

The default complete profiler makes this RMSNorm graded call window **5.42 times as long**. About
**81.5% of the profiled call time** is profiler overhead: 33.741 of 41.379 ms/case. A linear
projection is roughly 33.7 seconds per 1,000 similar cases.

## Where the time goes

`eval.metrics_plugin` establishes two boundaries for every test:

1. `record_axes`, immediately before the op, calls `flush_device_profiler` to discard input-prep
   programs.
2. The pytest call-hook exit calls `read_device_perf` to capture the op window and record
   `device_kernel_ns` and `device_num_cores`.

Both paths call `ttnn.ReadDeviceProfiler`, which finishes the command queue before reading.
Direct timers show:

| Work | Time/case | Share of 33.741 ms total profiler overhead |
|---|---:|---:|
| Device-profiler instrumentation during the test | 9.680 ms | 28.7% |
| Op-start flush (`ReadDeviceProfiler`) | 8.927 ms | 26.5% |
| End-of-test capture and reduction | 13.583 ms | 40.3% |
| Remaining plugin/order variance | 1.551 ms | 4.6% |

The two profiler reads directly consume **22.510 ms/case**, or 66.7% of total profiler overhead.
Almost all of that is inside `ReadDeviceProfiler`: its measured portions were 8.919 ms for the
start flush and 13.515 ms for the final read. `get_latest_programs_perf_data` and its Python scan
took only **0.018 ms/case**.

The C++ postprocessor also creates and emits a 10-bucket kernel-duration histogram on every read.
Setting `TT_METAL_LOGGER_LEVEL=error` saved 1.036 ms/case, but the complete profiler still added
32.706 ms/case. `LOGURU_LEVEL` does not suppress this C++ logging. The primary table uses the
default eval-runner behavior, which does not set `TT_METAL_LOGGER_LEVEL`.

The main optimization target is therefore the two synchronous device reads, especially the
pre-op flush. For RMSNorm's interleaved input path, `from_torch` dispatches no prep program, so the
flush is redundant. Other golden ops can perform real input-prep programs, so removing it globally
would mix prep time into their op metric. A lower-overhead design needs a cheap program-window
cursor or test marker, followed by one batched/device read, rather than a full device flush at both
boundaries.

## Interaction with graded-run L1 profiling

The complete perf profiler and complete cache-off L1 profiler were then enabled together for three
more 50-case runs:

| Mode | Mean/case | Increment |
|---|---:|---:|
| Complete perf profiler | 41.379 ms | -- |
| Complete perf profiler + graded cache-off L1 | 41.928 ms | **+0.549 ms / +1.33%** |

When the perf profiler is already enabled, adding the L1 measurement is small relative to the
existing profiler cost. The combined result is 34.291 ms/case above a profiler-off, L1-off graded
run.

## Controls and reproduction

The four eval profiler variables matched `eval_test_runner.sh`:

```text
TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1
TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1
```

The complete mode loaded `eval.metrics_plugin`, which exercised the production `record_axes`
flush and call-hook capture. The off mode also loaded that plugin, with the profiler environment
unset, so ordinary metrics-plugin overhead was present on both sides. `perf_timing_benchmark.py`
wrapped `ReadDeviceProfiler`, `get_latest_programs_perf_data`, and both production reducer entry
points without changing their results.

Kernel ccache variables were removed in every pass. No JIT-server endpoint was available, so the
profiler-compatible persistent cache was warmed with local up-front precompile. The measured graded
processes were fully warm; compilation location does not affect these call-hook timings.

Example complete-profiler command:

```bash
env -u TT_METAL_CCACHE_KERNEL_SUPPORT -u CCACHE_DISABLE -u CCACHE_DIR \
  TT_METAL_DEVICE_PROFILER=1 \
  TT_METAL_PROFILER_MID_RUN_DUMP=1 \
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
  TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 \
  PYTHONPATH="$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen" \
  PERF_TIMING_MODE=full PERF_TIMING_OUTPUT=/tmp/perf_timing.jsonl \
  scripts/run_safe_pytest.sh --no-precompile \
  tests/ttnn/unit_tests/base_functionality/graph_l1_traces/test_rms_norm_graded_l1_timing.py \
  -qq -p eval.metrics_plugin -p tests.plugins.perf_timing_benchmark
```
