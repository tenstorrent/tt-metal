# Real-time performance-profiler overhead

Measured on 2026-09-08 on Blackhole, from commit
`9b517bde6c921cbc7c8cec8042e829eafac38e82`. This evaluates the experimental
program real-time profiler as a replacement for the standard eval device profiler measured in
`PERF_PROFILER_OVERHEAD.md`.

## Result

The workload is the same 50-distinct-program RMSNorm matrix used by the other timing experiments:
FLOAT32 tiled `[1, 1, 32, W]`, with `W=32..1600` in steps of 32. Each mode ran in a fresh graded
pytest process with the persistent kernel cache already warm. There is no repeated program within a
process, so the program cache does not reuse an RMSNorm program. The table reports the median
aggregate call-hook time across three balanced runs per mode (`off, stream, stream, off, off,
stream`).

| Mode | 50-call time | Mean/case | Delta from off |
|---|---:|---:|---:|
| Profiler callback off | 368.368 ms | 7.367 ms | -- |
| One session-wide real-time callback | 371.104 ms | 7.422 ms | **+0.055 ms / +0.74%** |
| Standard eval profiler (previous experiment) | 2,068.947 ms | 41.379 ms | +33.741 ms / +441.8% vs its control |

The real-time profiler's observed increment is about **55 us/case**, or roughly **600 times smaller**
than the standard profiler's 33.741 ms/case overhead. Pairing each profiled run with its neighboring
control gives an even smaller 29 us/case average increment; this difference is below the normal
run-to-run spread, so 55 us is an observed result rather than a stable lower-level cost estimate.

All three streaming runs received exactly 50 valid records for 50 cases, with zero drops. Every
record named only `rms_norm_reader.cpp`, `rms_norm_compute.cpp`, and `rms_norm_writer.cpp`. The
unprofiled execution therefore produced a complete one-program-per-case stream for this workload.

After adding `core_count` to the real-time record, a further 50-case run matched the standard
profiler's `core_count` **exactly in all 50 cases**. Both reported the same set of counts:
`1, 4, 5, 6, 7, 8, 9, 10, 11, 22`. The callback took 20.3 us/record in that run, compared with
22.4 us/record before the field was added, so the extra host metadata had no measurable cost.

## Where the time goes

One additional instrumented run measured the Python collector itself:

| Work | Measured cost |
|---|---:|
| Register callback and query active state | 0.055 ms once/session |
| Copy one Python record in the callback | 22.4 us/record |
| All 50 callbacks | 1.122 ms of consumer-thread CPU time |
| Check delivery at each test boundary | 4.4 us/case |
| Unregister callback | 0.005 ms once/session |

Each callback batch contained one record in this test. Callback CPU work overlaps the test thread,
so it must not be added directly to the 55 us/case wall-time increment. The architecture's own
microbenchmarks report about 0.09 us for the dispatch-side signal and about 0.42 us for one
device-to-host drain. There are no synchronous profiler dumps or C++ postprocessing passes.

The `wait_per_case` mode waited on a condition until record count reached the case count. It timed
out zero times and spent a median 4.4 us/case in the wait/check. RMSNorm's output conversion already
synchronizes the device, so the callback record was present when the hook reached the boundary.
This is much cheaper than the standard profiler's two `ReadDeviceProfiler` calls, which consumed
22.510 ms/case by themselves.

The streamed device duration was stable across runs: approximately 6.83 us/case mean, 5.36 us
median, and 13.45 us p95. A separate standard-profiler pass reported 6.70 us mean, 5.22 us median,
and 13.24 us p95. The aggregate differences are 1.4--2.7%, consistent with the two profilers
measuring the same program interval.

## Exact attribution without a profiler flush

`ProgramRealtimeRecord` arrives asynchronously, so using the callback's current list index at
`record_axes` is not exact: an input-preparation program may have already been enqueued but reach the
callback after the marker. The existing standard profiler avoids that race by synchronously
consuming everything at `record_axes`, which is the source of its expensive pre-op flush.

The native binding already exposes a better boundary:

```python
start_runtime_id = ttnn._ttnn.get_device_operation_id()  # in record_axes, before the op
# dispatch the op and perform the golden check/readback
end_runtime_id = ttnn._ttnn.get_device_operation_id()    # at pytest call-hook exit
```

The counter returns the next ID that will be assigned at enqueue. Records whose runtime IDs are in
`[start_runtime_id, end_runtime_id)` belong to the same window the current eval profiler intends to
measure. Input-preparation programs have earlier IDs even if their callback delivery is late. The
golden output readback normally synchronizes the operation, after which waiting for callback
delivery was only 4.4 us/case in this experiment.

For a production eval integration, register one callback after the first test device is constructed,
keep it registered across module-scoped device lifetimes, and copy every record immediately in the
callback. `record_axes` should save the start counter on the active pytest item; the call-hook exit
should save the end counter, wait for the corresponding records, reduce them, and write the metric.
Closing a device drains callback consumers, and the data collector reattaches existing callback
registrations when a later module creates a new device.

## Limitations to resolve

1. This investigation adds `core_count` to `ProgramRealtimeRecord`. It is derived before dispatch
   from the distinct logical cores targeted by the program's kernels and cached per unique program,
   so the eval pipeline can retain `device_num_cores` without enabling the standard profiler. It is
   host metadata joined to the timing record by runtime ID; it does not enlarge the device-to-host
   packet.
2. The streamed runtime ID is currently truncated to 16 bits (`#46103`). A long golden process can
   exceed 65,536 dispatched programs even if it has fewer test cases. Attribution must unwrap IDs
   in stream order or the record format must be widened before relying on the raw ID range.
3. A mesh emits one record per chip for a runtime ID. The reducer must choose and document its
   meaning. Maximum duration across chips represents critical-path latency; summing chip durations
   preserves the current standard eval reducer's behavior.
4. The real-time profiler is inactive on dispatch configurations without a host-accessible profiler
   core, including some ETH-dispatch setups. The plugin must query
   `IsProgramRealtimeProfilerActive()` after device construction and explicitly mark performance as
   unavailable instead of silently recording zero.
5. Registering or querying the profiler before a device exists segfaulted in this build. The API
   contract says the active-state query is safe after device construction, and the existing helper
   carries the same warning. Pytest session start is therefore too early; first `pytest_runtest_call`
   or a device-fixture hook is safe.

The recommended graded design is the session-wide real-time callback plus runtime-ID windows. It
removes the standard profiler environment variables and both per-case device dumps. L1 collection
can then run beside it: up-front `NO_DISPATCH` capture remains the complete primary L1 source, while
the cache-off graded capture can remain a sampled validation path.

## Controls and reproduction

The standard profiler variables were explicitly removed in real-time runs:

```text
TT_METAL_DEVICE_PROFILER
TT_METAL_PROFILER_MID_RUN_DUMP
TT_METAL_PROFILER_CPP_POST_PROCESS
TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES
```

Kernel ccache variables were also removed in every pass. No JIT-server endpoint was available in
this shell; prior local precompile populated the same persistent cache that the graded processes
used. Because all measured graded programs were warm, whether the warm pass compiled locally or on
the JIT server does not affect these call-hook timings.

Example streaming command:

```bash
env -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_PROFILER_MID_RUN_DUMP \
  -u TT_METAL_PROFILER_CPP_POST_PROCESS \
  -u TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES \
  -u TT_METAL_CCACHE_KERNEL_SUPPORT -u CCACHE_DISABLE -u CCACHE_DIR \
  PYTHONPATH="$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen" \
  RT_PERF_TIMING_MODE=stream RT_PERF_TIMING_OUTPUT=/tmp/rt_perf_timing.jsonl \
  TT_METAL_LOGGER_LEVEL=error \
  scripts/run_safe_pytest.sh --no-precompile \
  tests/ttnn/unit_tests/base_functionality/graph_l1_traces/test_rms_norm_graded_l1_timing.py \
  -qq -p tests.plugins.realtime_perf_timing_benchmark
```

`realtime_perf_timing_benchmark.py` also accepts `off` and `wait_per_case`. It is an opt-in
measurement plugin and is not loaded by the normal test suite.
