# Tensor Prefetcher MPFE weight benchmark

The Blackhole Device-side Tensor Prefetcher defaults to static GDDR MPFE
weights. In free-sender / NOC1-sender / ordinary-operation order, they are:

```text
0 / 1 / 5
```

The two senders in each bank rendezvous after every prefetch request. This
prevents one sender from running ahead and contending with the peer that gates
overall completion. The measured FF1 cost was indistinguishable from zero, while
the repeated-request contention benchmark improved by approximately 1–1.6%.

Optional idle weights enable a dynamic policy: the kernel starts idle, switches
to the active weights while processing each PREFETCH request, and restores the
idle weights afterward. Stopping the Tensor Prefetcher always restores all three
hardware weights to `0/0/0`.

## MPFE controls

The active weights and optional idle weights are Tensor Prefetcher configuration
overrides:

```cpp
tt::tt_metal::experimental::TensorPrefetcherConfig config{
    .free_sender_mpfe_weight = 0,
    .noc1_sender_mpfe_weight = 1,
    .ordinary_mpfe_weight = 5,
    .idle_free_sender_mpfe_weight = 0,
    .idle_noc1_sender_mpfe_weight = 0,
    .idle_ordinary_mpfe_weight = 0,
    .synchronize_senders = true,
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
    idle_free_sender_mpfe_weight=0,
    idle_noc1_sender_mpfe_weight=0,
    idle_ordinary_mpfe_weight=0,
    synchronize_senders=True,
)
```

Each value must be an integer from 0 through 7. The three active values default
to `0/1/5`. An omitted idle field (or Python `None`) inherits its corresponding
active value, so existing callers remain static. Changing configuration requires
stopping and restarting the prefetcher, but does not require another Metal/TTNN
build. Sender synchronization defaults on. It may be disabled for static
comparisons, but dynamic ordinary-operation weights require it because that MPFE
slot is shared by both senders.

## Benchmark controls

The benchmark runners accept environment variables as a process-launch
convenience:

```bash
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT=0
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT=1
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT=5
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_FREE_SENDER_WEIGHT=0
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_NOC1_SENDER_WEIGHT=0
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_ORDINARY_WEIGHT=0
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_SYNCHRONIZE_SENDERS=1
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

It covers the default, representative static weights, static `015` without
request synchronization, and dynamic idle `000` to active `015`. Every case
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

## Mixed Llama-8B FF1 and SDPA

The mixed benchmark queues a receiver-contiguous per-device Llama-8B TP2 FF1
weight (`4096x7168`), runs decode SDPA against DRAM-resident K/V while the
DRISCs prefetch, and then consumes FF1 from the GCB. The complete FF1 receiver
shard fits in the GCB even on a seven-bank harvested device, allowing prefetch
to complete and restore dynamic idle weights while SDPA is still generating
ordinary traffic. Context length changes the amount of ordinary K/V traffic:

```bash
PYTHONPATH=$PWD \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT=0 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT=1 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT=5 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_FREE_SENDER_WEIGHT=0 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_NOC1_SENDER_WEIGHT=0 \
TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_ORDINARY_WEIGHT=0 \
BENCH_SDPA_CONTEXT=1024 \
BENCH_TRACE_REPEATS=20 \
pytest -sv \
tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_mpfe_mixed_bench.py::test_mpfe_mixed_llama8b_ff1_sdpa
```

Run the focused randomized comparison with:

```bash
tests/scripts/single_card/run_bh_tensor_prefetcher_mpfe_mixed.py
```

It compares unsynchronized static `000`, unsynchronized static `015`,
synchronized static `015`, and synchronized dynamic `000` to `015` across
contexts 512, 1024, 2048, and 4096. Configure it with
`MPFE_MIXED_CONTEXTS`, `MPFE_MIXED_ITERATIONS`, `BENCH_TRACE_REPEATS`,
`MPFE_RANDOM_SEED`, and `OUTPUT_DIR`. Results include raw JSONL, `summary.csv`,
`paired-comparisons.csv`, the exact manifest, and pytest logs. The reports
separately measure active priority, sender synchronization, and dynamic idle
restoration; the final comparison holds active weights and synchronization
constant.

Dynamic priority can help only when ordinary work continues after a prefetch
request finishes. If prefetch occupies the entire SDPA interval, dynamic should
match static aside from its request-boundary register writes.

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
