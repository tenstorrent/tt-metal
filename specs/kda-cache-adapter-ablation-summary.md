# KDA cache-adapter ablation: short summary

## Question

What is the cost of converting PR7's native KDA prefill cache to and from the
K3 disaggregated-decode contract on SP1xTP8, SP2xTP4, and SP4xTP2?

## Experiment

- Real Kimi-K3 layer 1, `B=1`, `T=5120`, eight Blackhole devices, FABRIC_1D.
- Native S: `[1,Hlocal,128,128]`, FP32 TILE, interleaved DRAM.
- Contract S: ND-sharded DRAM with `[1,1,128,32]` shards (16,384 bytes).
- Native convolution: `[1,3,3*Hlocal*128]`, BF16 ROW_MAJOR, interleaved DRAM.
- Contract convolution: ND-sharded DRAM with `[1,3,64]` shards (384 bytes).
- Preallocated destinations; 20 synchronized samples × 100 trace replays.
- Correctness: independent CPU PCC plus bit-identical real and patterned
  native→contract→native round trips.
- Attribution: targeted Tracy run on the worst-case SP4xTP2 layout.

## Results

Adapter medians are in microseconds; layer medians remain in milliseconds.
Percentages are relative to the real KDA-layer median.

| Layout | Layer (ms) | Export S (µs) | Export conv (µs) | Export total (µs) | Import S (µs) | Import conv (µs) | Import total (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | 9.7171 | 12.5 | 27.4 | 35.4 (0.364%) | 11.6 | 36.2 | 44.9 (0.462%) |
| SP2xTP4 | 9.6486 | 16.3 | 49.0 | 61.7 (0.640%) | 16.2 | 69.7 | 82.6 (0.856%) |
| SP4xTP2 | 10.1103 | 23.6 | 94.3 | 114.8 (1.136%) | 23.8 | 133.9 | 154.9 (1.532%) |

All three layouts passed. Output PCC was 0.999902–0.999905, and both cache
components were bit-identical after round trip. Recommendation: use preallocated,
warmed adapters at the handoff boundary; the current KDA kernels do not support
a complete direct contract-layout path.

## Why S is faster despite moving more bytes

This result is controlled by kernel parallelism and transfer granularity, not
payload size:

1. S is tiled. `CopyDeviceOperation` selects `DefaultTilized`, partitions its
   768 tiles per SP4 device over the full worker grid, and Tracy reports 110
   active cores. Its pages are 4,096 bytes.
2. Convolution is row-major with logical shape `[1,3,W]`. `DefaultRowMajor`
   partitions work by logical row, so there are only three work items and Tracy
   reports only 3 active cores. Each core must redistribute 288 tiny 128-byte
   pages per row at SP4xTP2.
3. Consequently, SP4 Tracy measures median device-kernel time of 16.410 µs for
   S export versus 90.473 µs for convolution export, and 16.278 µs versus
   128.816 µs on import. The reverse row-major path is measurably worse, but the
   profile does not isolate a more specific cause inside its reader/writer.

At SP4, each device moves about 3.15 MB of S but only 0.11 MB of convolution
state. S nevertheless wins because 110 cores issue large tile transfers, while
convolution is bottlenecked by a three-core, many-small-page redistribution.
The most relevant follow-up optimization is therefore a row-major ND copy path
that parallelizes page groups independently of the tensor's three logical rows.

## Evidence

- Full report: `specs/kda-cache-adapter-ablation-results.md`
- Raw samples: `specs/kda-cache-adapter-ablation-results.jsonl`
- Full safe-test log: `specs/kda-cache-adapter-ablation-safe.log` (local)
- Tracy CSV: `generated/profiler/reports/2026_08_25_22_32_56/ops_perf_results_2026_08_25_22_32_56.csv` (local)
- Factory selection: `copy_device_operation.cpp:65-73`
- Row-major core split: `copy_default_row_major_program_factory.cpp:70-73`
- Tiled core split: `copy_default_tilized_program_factory.cpp:47-64`
