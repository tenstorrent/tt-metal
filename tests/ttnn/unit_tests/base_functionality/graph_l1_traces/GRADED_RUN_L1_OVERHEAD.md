# Graded-run L1 profiling overhead

Measured on 2026-09-08 on Blackhole, from commit `8458ac2a57ab176546f9df3181370bf1f102753e`.
The question is how much a golden-test case slows down when its real graded call runs with the
in-memory program cache disabled and a `RunMode.NORMAL` graph capture is reduced to L1 peaks.

## Result for 50 distinct RMSNorm programs

The workload is `test_rms_norm_graded_l1_timing.py`: 50 valid RMSNorm golden calls with
`[1, 1, 32, W]` FLOAT32 tiled inputs, `W=32..1600` in steps of 32. Every case runs the normal
golden reference, tensor creation, device operation, readback, and output check. The baseline
finished with 50 program-cache entries, confirming that the 50 shapes had 50 distinct program
hashes. Each mode was run three times; the table reports the median aggregate time from the
per-test call hook.

| Graded mode | 50-call time | Mean/case | Delta from baseline |
|---|---:|---:|---:|
| Program cache on, no capture | 375.095 ms | 7.502 ms | -- |
| Program cache off, no capture | 384.468 ms | 7.689 ms | +9.373 ms / +2.50% |
| Program cache off, capture only | 405.935 ms | 8.119 ms | +30.840 ms / +8.22% |
| Program cache off, capture + current C++ reducer | 434.122 ms | 8.682 ms | **+59.027 ms / +15.74%** |

For this mostly-unique workload, adding the complete L1 measurement to the graded run costs
**1.181 ms per case**. Of that, disabling the program cache costs **0.187 ms/case** and L1 graph
capture plus reduction costs **0.993 ms/case** on top of the cache-off run. A linear projection is
about 1.18 seconds per 1,000 RMSNorm-like cases. Trace size and operation complexity affect that
projection.

All 150 full-profile cases produced nonzero L1 results. Captures contained a median of 44 nodes
(range 44--47). The baseline reached 50 program-cache entries; every cache-off mode remained at
zero.

## Where profiler time goes

Direct timers inside the complete profiler give this median breakdown:

| Profiler work | Time/case | Share of the observed 0.993 ms profiler delta |
|---|---:|---:|
| Begin C++ graph capture | 0.005 ms | 0.5% |
| Emit graph events during the body (estimated by capture body minus cache-off body) | 0.046 ms | 4.6% |
| End capture and convert its JSON trace to Python | 0.377 ms | 38.0% |
| `extract_resource_usage_per_core` | 0.480 ms | 48.3% |
| Trial-order/noise remainder | 0.085 ms | 8.6% |

Only about 0.05 ms/case is spent installing the capture and emitting events. Most measured
profiler time is trace conversion. `end_graph_capture` materializes the nlohmann JSON trace as
Python objects. The existing reducer binding then calls Python `json.dumps`, reparses that string
into nlohmann JSON, and finally scans it in C++.

An equivalent direct Python scan reduced the live 44-node RMSNorm traces in **0.025 ms/case**
instead of 0.480 ms/case. With that scan, the complete measured mode was 8.235 ms/case:
0.545 ms/case above program-cache-off and 0.733 ms/case (+9.77%) above baseline. Offline
microbenchmarks showed the same bridge cost:

| Saved trace | Nodes | Current reducer binding | Direct Python scan |
|---|---:|---:|---:|
| Matmul | 24 | 0.192 ms | 0.006 ms |
| Conv2d | 186 | 1.622 ms | 0.041 ms |

For a production graded-run profiler, a better API would end the capture and compute the five
peaks in C++, returning only the peak values. That avoids both full-trace conversions. Retaining
the raw trace should be an opt-in diagnostic for invalid or outlier cases.

## Cache-hit-heavy comparison

The same four modes were run three times over the 15 RMSNorm regression cases repeated 20 times
(300 calls). That suite has only three distinct program hashes, so the baseline benefits from 297
in-memory cache hits while cache-off rebuilds every program. Median call-hook totals were:

| Graded mode | 300-call time | Mean/case | Delta from baseline |
|---|---:|---:|---:|
| Program cache on, no capture | 970.076 ms | 3.234 ms | -- |
| Program cache off, no capture | 1,120.713 ms | 3.736 ms | +0.502 ms/case |
| Program cache off, capture only | 1,379.219 ms | 4.597 ms | +1.364 ms/case |
| Program cache off, capture + current reducer | 1,514.242 ms | 5.047 ms | **+1.814 ms/case / +56.10%** |

This is the upper-risk pattern for a normal-pass design: turning off the program cache removes real
reuse, and the tests themselves are short enough that a roughly 1 ms profiler cost is a large
percentage. It is less representative of the expected golden matrix, but it explains why the
default should remain up-front collection and why normal-pass profiling should be selectable.

## Controls and reproduction

The initial warm pass collected and compiled all 50 programs before the graded process:

```text
50 ops stashed across 50 bodies -> 50 unique programs
compiled 50 programs in 14.2s (workers=32, errors=0)
```

No JIT-server endpoint was configured in this shell, so this warm pass compiled locally. This is a
limitation of the test environment, not a cold graded-run measurement: each subsequent graded
process reported **258/258 persistent JIT-cache hits** and issued no compilation. A JIT server
changes where the warm-pass `CompileProgram` work occurs; it does not change the already-warm
graded path measured here.

Kernel ccache variables were removed for the warm pass and every measurement:
`TT_METAL_CCACHE_KERNEL_SUPPORT`, `CCACHE_DISABLE`, and `CCACHE_DIR`. Each graded mode ran through
`scripts/run_safe_pytest.sh` in its own process and used the RMSNorm suite's module-scoped device.
Program cache policy was applied around each pytest call and restored afterward.

Warm the workload (omit `--no-jit-server` and configure `TT_METAL_JIT_SERVER_ENDPOINT` to use the
normal JIT-server route):

```bash
env -u TT_METAL_CCACHE_KERNEL_SUPPORT -u CCACHE_DISABLE -u CCACHE_DIR \
  PYTHONPATH="$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen" \
  L1_TIMING_MODE=baseline L1_TIMING_OUTPUT=/tmp/l1_timing.jsonl \
  scripts/run_safe_pytest.sh --precompile --no-jit-server \
  tests/ttnn/unit_tests/base_functionality/graph_l1_traces/test_rms_norm_graded_l1_timing.py \
  -qq -p tests.plugins.l1_timing_benchmark
```

Then run `baseline`, `pc_off`, `capture`, and `full` three times each with `--no-precompile`,
preserving the same `TT_METAL_CACHE`:

```bash
env -u TT_METAL_CCACHE_KERNEL_SUPPORT -u CCACHE_DISABLE -u CCACHE_DIR \
  PYTHONPATH="$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen" \
  L1_TIMING_MODE=full L1_TIMING_OUTPUT=/tmp/l1_timing.jsonl \
  scripts/run_safe_pytest.sh --no-precompile \
  tests/ttnn/unit_tests/base_functionality/graph_l1_traces/test_rms_norm_graded_l1_timing.py \
  -qq -p tests.plugins.l1_timing_benchmark
```

The plugin calls the C++ graph-capture binding directly because L1 accounting does not need the
public report wrapper's Python argument recording. This measures an efficient L1-only graded-run
implementation. A full graph report with Python I/O metadata has additional overhead.
