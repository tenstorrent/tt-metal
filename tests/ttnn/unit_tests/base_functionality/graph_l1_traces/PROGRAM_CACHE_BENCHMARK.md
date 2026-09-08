# Program-cache miss benchmark for L1 profiling

Measured on 2026-09-08 at commit `edf6276af5a` on Blackhole. The workload is 50 synchronized
matmuls with `A=[1,1,32,K]`, `B=[1,1,K,32]`, `K=32..1600` in steps of 32, alternating BF16 and
BFLOAT8_B. Inputs are prepared before the timed region. The program-cache-on runs assert that all
50 cases create distinct program-cache entries.

This experiment answers the practical question raised by graph-capture L1 profiling: if
`RunMode.NORMAL` needs the program cache disabled to report every program's circular buffers, how
much does that slow a golden-test-like matrix where nearly every case has a distinct program?

LayerNorm was the intended example, but this build does not export `ttnn.layer_norm`; its
normalization module currently binds only `batch_norm`. Matmul was selected because its cache key
includes both input tensor specs, allowing the workload to verify 50 distinct hashes directly.

## Controls

- Every process unsets `TT_METAL_CCACHE_KERNEL_SUPPORT`. The benchmark asserts that it is absent.
  The runtime enables kernel ccache based on presence, so setting it to `0` would still enable it.
- `TT_METAL_LOG_KERNEL_COMPILE=1`, `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`, and
  `LOGURU_LEVEL=DEBUG` retain every compilation phase, full compile/link command, per-kernel
  completion time, cache-hit message, and final JIT telemetry summary.
- The emitted commands were audited; none invokes `ccache`.
- `scripts/run_safe_pytest.sh --no-precompile` prevents the test wrapper from warming the cache.
- Cold runs use separate initially empty `TT_METAL_CACHE` directories.
- The completed cold-on cache was copied byte-for-byte to separate warm-on and warm-off directories
  before either warm measurement.
- Each operation is synchronized before its timing ends. Input creation is outside the timed
  region, matching a test session whose fixtures have already prepared its operands.

The exact command shape was:

```bash
env -u TT_METAL_CCACHE_KERNEL_SUPPORT -u CCACHE_DISABLE -u CCACHE_DIR \
    TT_METAL_CACHE=<fresh-or-warmed-directory> \
    L1_BENCH_PROGRAM_CACHE=<on-or-off> \
    L1_BENCH_NUM_CASES=50 \
    LOGURU_LEVEL=DEBUG \
    TT_METAL_LOG_KERNEL_COMPILE=1 \
    TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
    scripts/run_safe_pytest.sh --no-precompile \
    tests/ttnn/unit_tests/base_functionality/graph_l1_traces/test_distinct_program_cache_benchmark.py \
    -q -s
```

## Results

| Persistent kernel cache | Program cache | 50-op time | Mean/op | Median/op | JIT cache |
|---|---:|---:|---:|---:|---:|
| Cold | On  | 34.843431 s | 696.758 ms | 695.053 ms | 0/258 hits |
| Cold | Off | 35.035686 s | 700.606 ms | 699.008 ms | 0/258 hits |
| Warm | On  | 0.164132 s | 3.240 ms | 3.193 ms | 258/258 hits |
| Warm | Off | 0.171028 s | 3.378 ms | 3.342 ms | 258/258 hits |

Disabling the program cache added 0.192255 s over 50 cold cases (3.845 ms/case, 0.55%) and
0.006897 s over 50 disk-warm cases (0.138 ms/case, 4.20%). The warm relative percentage is larger
because the baseline is only 3.24 ms/case; its absolute difference is about 139 microseconds per
case.

The paired case measurements favored program cache on in 28/50 cold cases and 36/50 warm cases.
Cold compilation noise is therefore larger than much of the cache-management difference. These
runs support the conclusion that disabling the program cache has little throughput cost for a
mostly-unique test matrix. Assuming linear scaling, the observed difference is about 3.85 seconds
per 1,000 cold-compiling cases or 0.14 seconds per 1,000 disk-warm cases.

This is one controlled 50-case run per condition, so the percentages should be treated as scale
estimates rather than precise long-run constants. The absolute result is clear enough for the
design decision: disabling the program cache does not materially change the cost of a test matrix
dominated by distinct kernel compilation.

## Compilation audit

| Condition | Compile commands | Link commands | Compiled targets | `ccache` commands | Program entries |
|---|---:|---:|---:|---:|---:|
| Cold/on | 258 | 258 | 258 | 0 | 50 |
| Cold/off | 258 | 258 | 258 | 0 | 0 |
| Warm/on | 0 | 0 | 0 | 0 | 50 |
| Warm/off | 0 | 0 | 0 | 0 | 0 |

The cold-on and cold-off runs compiled identical target-name/hash sets. The 258 targets comprise
eight dispatch targets plus five matmul targets for each of the 50 programs. Both warm runs hit all
258 persistent-cache entries and emitted no compile or link command.

## Interpretation for the L1 profiler

For a case matrix with few program-cache hits, the program cache saves little work: every case
still builds and compiles a new program. With the cache disabled, completed program objects can be
destroyed instead of retained; the measured extra lifecycle work is small compared with cold JIT
compilation and remains sub-millisecond when kernel binaries are already on disk.

This supports solution A in `L1_profiling_critical_findings.md` for golden-test-style profiling:
disable and clear the program cache, capture the real call in `RunMode.NORMAL`, and attribute the
whole-trace peak to the operation frame. Workloads with substantial program reuse still need a
separate measurement because this experiment deliberately contains no hits.
