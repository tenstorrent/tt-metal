# Gemma4 GPU KV PCC performance

## Test configuration

The mock 256K test uses Gemma4-31B-it on one Blackhole 8×4 mesh, with six allocated KV slots and one GPU comparison of slot 0. The prompt is 262144 captured Gutenberg tokens, processed in 32 chunks of 8192 tokens. Validation covers all 60 layers and all 1680 PCC scores.

The measured host is an AMD EPYC 9354P with 32 cores/64 threads. PyTorch uses 16 OpenMP/MKL threads. Weight caches are on local storage; weight and kernel caches are warm. Filesystem caches are not cleared.

The GPU reference is:

```text
/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135
```

Use the [test environment](PREFILL_TEST_FLOWS.md#run-the-tests), then run:

```bash
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
/usr/bin/time -p pytest \
    'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-256k]' \
    -sv --basetemp=/tmp/gemma4-pcc
```

## Time breakdown

The test passes with minimum PCC **0.916626**, above the **0.91** threshold.

| Phase | Time |
| --- | ---: |
| Python/pytest startup and runner imports | 9.3 s |
| Device/fabric initialization, model config and KV allocation | 7.6 s |
| Cached weight loading and model construction | 54.9 s |
| Compile warmup and H2D/completion-service setup | 20.5 s |
| Migration table build and export, first call | 56.0 s |
| Migration table build and export, second call | 54.2 s |
| Trace capture and completion warmup | 2.1 s |
| Producer startup, prompt loading and first input | 13.0 s |
| Prefill execution and chunk delivery, 32 chunks | 14.4 s |
| Producer completion, table import and device-map loading | 13.7 s |
| GPU reference loading and conversion | 48.6 s |
| TT readback, host gathering and address checks | 1m 29.4s |
| PCC and finite-value checks | 1m 15.3s |
| Validation logging and JSON report writing | 0.02 s |
| Service cleanup, trace release and mesh closure | 0.68 s |
| Device-driver teardown, process destruction and pytest exit | 7.2 s |
| **Total elapsed** | **7m 46.8s** |

Validation takes **3m 33.2s**. The table covers the complete pytest process lifetime. Setup and teardown intervals use log timestamps; validation uses `perf_counter` phase timers. Values are rounded. Weight loading includes model construction; compilation includes a warmup forward pass and service creation.

## Validation work

1. Load one GPU reference layer in validation channel order and convert its BF16 values to FP32.
2. Slice slot 0's populated prefix on the owning mesh, untilize a temporary copy to BF16 row-major, and read it through the TTNN command queue.
3. Copy host shards into head and token order. Check the first and last populated 32-token block on each CP rank against the migration table using UMD reads.
4. Compute one PCC per sliding K/V head, and separate rotary-K and V scores per global head.

PCC uses an **8 MiB reusable FP32 buffer**. Each block computes three dot products for the two sums of squares and the cross-product. Block means and centered statistics combine into one whole-head correlation. Nonfinite inputs propagate into the statistics and fail validation. TT readback stays BF16 until copied into this buffer.

The reader gathers one layer's K, V, or packed KV tensor at a time. It does not retain the entire model's host KV. Address checks cover **26,240 blocks**, totaling **225.8 MiB**, in addition to the full-prefix GPU comparison.

## Data volume and throughput

These are cumulative logical tensor bytes, excluding file headers and address-table samples. The prepared reference uses 60 files at every context length; the loader selects the requested prefix.

| Context | GPU BF16 reference | TT BF16 readback | GPU reference converted to FP32 |
| --- | ---: | ---: | ---: |
| 8K | 6.641 GiB | 6.641 GiB | 13.281 GiB |
| 16K | 13.281 GiB | 13.281 GiB | 26.563 GiB |
| 128K | 106.25 GiB | 106.25 GiB | 212.5 GiB |
| 256K | 212.5 GiB | 212.5 GiB | 425 GiB |

At 256K:

- Each of 50 sliding layers contains 4 GiB of BF16 K/V.
- Each of 10 global layers contains 1.25 GiB of BF16 packed KV.
- Reference loading and conversion process 212.5 GiB in **48.6 s**, or **4.38 GiB/s**, including filesystem-cache effects and FP32 conversion.
- TT readback, host gathering, and address checks take **89.4 s**. This includes host processing and is not a device-link bandwidth measurement.
- PCC checks 1680 scores in **75.3 s**, or **22.32 scores/s**.

## Migration address table

The table describes the full allocated capacity of all six slots:

```text
Sliding: 50 layers × 16 heads × 2 tensors = 1,600 combinations
Global:  10 layers ×  4 heads × 1 tensor  =    40 combinations

1,640 × (262,144 tokens ÷ 32) × 6 slots = 80,609,280 populated entries
```

The protobuf file is **1.69 GiB**. Each entry maps a layer, head, slot, and 32-token block to a device, DRAM address, and byte count. KV values stay in device memory.

Two mock-mode branches in the shared runner independently build and export this table, taking **56.0 s + 54.2 s**. These intervals include address generation, table population, serialization, and disk writes. Producer completion, table import, and device-map loading take **13.7 s** before validation.

## Logs and reports

Each pytest case writes `runner.log`, `producer.log`, `table.pb`, `device_map.json`, and `gemma4_slot0.json` under its temporary directory. The JSON report contains every layer/head score, cache-type minima, and total and per-layer validation timings. Each completed layer logs reference-loading, readback, and PCC time.

The producer's `DONE wall=... throughput=...` line measures scheduling and input delivery. Use the full pytest elapsed time for test duration and `gemma4_slot0.json` for validation costs.
