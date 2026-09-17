# Tensor Prefetcher MPFE weight benchmark

The Blackhole Device-side Tensor Prefetcher holds static GDDR MPFE weights from
startup until shutdown. Defaults, in free-sender / NOC1-sender /
ordinary-operation order, are:

```text
0 / 1 / 5
```

The two senders in each bank rendezvous after every prefetch request. This
prevents one sender from running ahead and contending with the peer that gates
overall completion. The measured FF1 cost was indistinguishable from zero, while
the repeated-request contention benchmark improved by approximately 1–1.6%.

Stopping the Tensor Prefetcher restores all three hardware weights to `0/0/0`.

## MPFE controls

The weights are optional Tensor Prefetcher configuration overrides:

```cpp
tt::tt_metal::experimental::TensorPrefetcherConfig config{
    .free_sender_mpfe_weight = 0,
    .noc1_sender_mpfe_weight = 1,
    .ordinary_mpfe_weight = 5,
};
tt::tt_metal::experimental::StartTensorPrefetcher(mesh_device, config);
```

Python callers can pass the equivalent keyword arguments:

```python
ttnn.experimental.start_tensor_prefetcher(
    device,
    free_sender_mpfe_weight=0,
    noc1_sender_mpfe_weight=1,
    ordinary_mpfe_weight=5,
)
```

Each value must be an integer from 0 through 7. The weights are fixed from
startup until shutdown; changing them requires stopping and restarting the
prefetcher, but does not require another Metal/TTNN build. Omitted fields (or
Python `None`) select the defaults.

## Benchmark controls

The benchmark runners accept environment variables as a process-launch
convenience:

```bash
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT=0
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT=1
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT=5
```

The Python benchmark helper validates set values and passes them explicitly to
`start_tensor_prefetcher`; unset values are omitted so the defaults apply. The
Tensor Prefetcher implementation does not read benchmark environment
variables.

## Validation and contention

After building, run the compact sanity matrix:

```bash
tests/scripts/single_card/run_bh_tensor_prefetcher_mpfe_sanity.sh
```

It covers the default, `000`, `014`, `037`, and `777`. Every case
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
