# Tensor Prefetcher MPFE policy benchmark

The Device-side Tensor Prefetcher reads its benchmark policy when it starts.
Build Metal/TTNN once after changing the C++ or DRISC sources. Policy and weight
changes after that require only a fresh test process; they do not require another
build.

## Policy controls

Set `TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY` to one of:

| Policy | Idle (sender 0 / sender 1 / ordinary) | While prefetching |
| --- | --- | --- |
| `dynamic-007` (default) | `H/H/H` | `0/0/H` |
| `dynamic-000` | `0/0/0` | `0/0/H` |
| `static-000` | `0/0/0` | `0/0/0` |
| `static-777` | `H/H/H` | `H/H/H` |
| `static-007` | `0/0/H` | `0/0/H` |
| `static-037` | `0/M/H` | `0/M/H` |
| `static-770` | `H/H/0` | `H/H/0` |

`H` defaults to 7. Override it with
`TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT=0..7`. For example,
`dynamic-000` with `HIGH_WEIGHT=5` uses `0/0/0` while idle and `0/0/5`
while prefetching.

`M` defaults to 3 and is controlled by
`TT_METAL_BENCHMARK_TENSOR_PREFETCHER_MEDIUM_WEIGHT=0..7`. In `static-037`,
the first sender is the free subchannel, the second sender is the NOC1-endpoint
subchannel, and the final slot is ordinary-operation traffic.

The older `TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ACTIVE_WEIGHT=0..7` sweep
remains available with `dynamic-007`. Do not combine it with another policy.

For arbitrary experiments without adding another named policy, set both generic
tuples in free-sender/NOC1-sender/ordinary order:

```bash
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_WEIGHTS=0,2,5
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ACTIVE_WEIGHTS=0,3,7
```

Tuple overrides cannot be combined with a named policy or individual weight
settings.

Stopping the Tensor Prefetcher restores all three hardware weights to `0/0/0`,
regardless of the selected benchmark policy.

## Contention test

Run each configuration in a fresh process from the repository root:

```bash
PYTHONPATH=$PWD TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=dynamic-000 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT=5 \
BENCH_TRACE_REPEATS=50 \
pytest -sv tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention
```

After building, run the compact sanity matrix before collecting performance
numbers:

```bash
tests/scripts/single_card/run_bh_tensor_prefetcher_mpfe_sanity.sh
```

It runs the production default, every named policy, a generic tuple, and a
forced-synchronization baseline in separate processes. Each case performs
initial and final byte validation and verifies that the Tensor Prefetcher stops cleanly.
Set `BENCH_TRACE_REPEATS` to change the default two replays, or append pytest
arguments such as `--timeout=60`.

Sweep the most useful comparison matrix:

```bash
for policy in static-000 static-777 static-007 static-037 dynamic-007 dynamic-000; do
  for high in 3 5 7; do
    PYTHONPATH=$PWD \
    TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=$policy \
    TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT=$high \
    BENCH_TRACE_REPEATS=50 \
    pytest -sv tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention
  done
done
```

Run the static middle ground directly with independently configurable medium
and high weights:

```bash
PYTHONPATH=$PWD \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=static-037 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_MEDIUM_WEIGHT=3 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT=7 \
BENCH_TRACE_REPEATS=50 \
pytest -sv tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention
```

The test reports the effective idle and active tuples, total elapsed time,
Tensor Prefetcher bandwidth, ordinary-read bandwidth, and combined bandwidth.

Set `TT_METAL_BENCHMARK_RESULT_JSONL` to append the same metrics as one JSON
object per run:

```bash
TT_METAL_BENCHMARK_RESULT_JSONL=$PWD/generated/mpfe-results.jsonl \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=dynamic-000 \
BENCH_TRACE_REPEATS=50 \
pytest -sv tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention
```

Run the complete policy and weight matrix repeatedly with:

```bash
BENCH_SUITE_ITERATIONS=3 BENCH_TRACE_REPEATS=50 \
tests/scripts/single_card/run_bh_tensor_prefetcher_mpfe_benchmarks.sh
```

The runner creates a timestamped directory under `generated/` containing
`benchmark.log` and `results.jsonl`. Its default matrix covers high weights
3/5/7, valid medium weights 1/3/5, active weights 1/3/5/7, all named policies,
the production default, and the forced-sync control. Override
`MPFE_HIGH_WEIGHTS`, `MPFE_MEDIUM_WEIGHTS`, `MPFE_ACTIVE_WEIGHTS`, or
`OUTPUT_DIR` as needed.

To isolate the cost of the per-request sender synchronization used by
`dynamic-000`, compare `static-007` with and without forced synchronization.
Both runs use identical weights, so their difference measures synchronization
overhead rather than MPFE weighting:

```bash
for force_sync in 0 1; do
  TT_METAL_BENCHMARK_RESULT_JSONL=$PWD/generated/mpfe-sync-overhead.jsonl \
  TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=static-007 \
  TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FORCE_REQUEST_SYNC=$force_sync \
  BENCH_TRACE_REPEATS=50 \
  pytest -sv tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention
done
```

## End-to-end matmul test

The operation-level end-to-end benchmark measures real Tensor
Prefetcher-to-matmul overlap. It currently requires an unharvested eight-bank
Blackhole because its matmul receiver topology is the production 64-core
layout:

```bash
PYTHONPATH=$PWD \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=dynamic-000 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT=5 \
BENCH_TRACE_REPEATS=100 \
pytest -sv \
tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bench.py::test_bench_dram_core_repeats_recv_contig \
-k '1B_FF1 and shard_contiguous'
```

These environment variables apply to every workload that calls
`ttnn.experimental.start_tensor_prefetcher`, so the same policy matrix can also
wrap a full-model performance command. Keep the model command and all other
settings identical, use a fresh process for each policy, and compare the model's
device throughput or latency rather than the synthetic bandwidth alone.
