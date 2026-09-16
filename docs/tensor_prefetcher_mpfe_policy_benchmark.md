# Tensor Prefetcher MPFE weight benchmark

The Blackhole Device-side Tensor Prefetcher holds static GDDR MPFE weights from
startup until shutdown. Production defaults, in free-sender / NOC1-sender /
ordinary-operation order, are:

```text
0 / 1 / 5
```

Stopping the Tensor Prefetcher restores all three hardware weights to `0/0/0`.

## Benchmark controls

Override any weight for benchmark experiments:

```bash
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT=0
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT=1
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT=5
```

Each value must be an integer from 0 through 7. The Tensor Prefetcher reads the
variables when it starts, so each configuration must run in a fresh process.
Changing weights does not require another Metal/TTNN build.

## Validation and contention

After building, run the compact sanity matrix:

```bash
tests/scripts/single_card/run_bh_tensor_prefetcher_mpfe_sanity.sh
```

It covers the production default, `000`, `014`, `037`, and `777`. Every case
performs initial and final byte validation and verifies clean shutdown.

Run one contention measurement directly:

```bash
PYTHONPATH=$PWD \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT=0 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT=1 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT=5 \
BENCH_TRACE_REPEATS=50 \
pytest -sv \
tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention
```

Set `TT_METAL_BENCHMARK_RESULT_JSONL` to append machine-readable metrics.

## End-to-end matmul

The receiver-contiguous benchmark measures real Tensor-Prefetcher-to-matmul
overlap. It uses every available Blackhole DRAM bank, keeps eight receivers per
bank, and pads the model shape for the resulting ring. An unharvested device
retains the production scattered 64-core topology; a harvested device uses a
compact logical receiver grid.

```bash
PYTHONPATH=$PWD \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT=0 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT=1 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT=5 \
BENCH_TRACE_REPEATS=100 \
pytest -sv \
tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bench.py::test_bench_dram_core_repeats_recv_contig \
-k '3B_FF1 and shard_contiguous'
```

## Static-weight optimization

Run the adaptive optimizer when model workload or device topology changes:

```bash
tests/scripts/single_card/run_bh_tensor_prefetcher_mpfe_optimize.py
```

The optimizer:

1. Runs five randomized passes over all 36 tuples `0/M/H`, where
   `0 <= M <= H <= 7`.
2. Selects the six leaders and runs fifteen randomized confirmation passes.
3. Brackets every pass with `static-000` and linearly interpolates between
   those sentinels to compensate for performance drift.

Outputs include `stage1-tuning-ranking.csv`,
`stage2-confirmation-ranking.csv`, `results.jsonl`, `summary.log`, and a
separate `pytest.log`. Interrupted runs can resume with the same `OUTPUT_DIR`;
the manifest prevents mixing configurations or device topologies.

Configure the run with:

- `MPFE_MATMUL_SHAPE` (default `3B_FF1`)
- `MPFE_TUNING_ITERATIONS` (default `5`)
- `MPFE_CONFIRM_ITERATIONS` (default `15`)
- `BENCH_TRACE_REPEATS` (default `100`)
- `MPFE_RANDOM_SEED`
- `OUTPUT_DIR`
