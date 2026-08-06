# Stage: 02-sdpa-chunk-192

- source commit: `0ef4d8efc93`
- kernel time (mean of replays 2-10): **24.445 ms**
- change from the previous stage: **-1.729 ms**
- device ops: **345**
- note: SDPA chunked 192 so 576 divides evenly.

## What this change was

**SDPA chunk 256 → 192.** 576 tokens divide into exactly three chunks of 192. The previous 256 left a third chunk only a
quarter full and paid for the padding. **182 → 111 us per op.**

The profiler pointed straight at this: SDPA was 14.9% of baseline at 182.6 us per instance, and
`576 / 256 = 2.25` is visibly not an integer.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 146.41 | 14.495 | 59.3 |
| BinaryNgDeviceOperation | 124 | 28.66 | 3.554 | 14.5 |
| SDPAOperation | 24 | 110.43 | 2.650 | 10.8 |
| LayerNormDeviceOperation | 49 | 32.36 | 1.585 | 6.5 |
| NlpCreateHeadsDeviceOperation | 24 | 64.38 | 1.545 | 6.3 |
| NLPConcatHeadsDeviceOperation | 24 | 20.71 | 0.497 | 2.0 |
| UnaryDeviceOperation | 1 | 122.78 | 0.123 | 0.5 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 24 | 273.6 | 6.566 | 48 | 17.9 | 18.1 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 128.3 | 3.079 | 48 | 28.6 | 29.8 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 125.1 | 3.002 | 48 | 39.2 | 39.7 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 47.2 | 1.133 | 48 | 26.0 | 32.8 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 489.3 | 0.489 | 48 | 80.1 | 30.5 | HiFi4 |
| 576 x 1024 x 4096 | 1 | 181.4 | 0.181 | 48 | 54.0 | 27.3 | HiFi4 |
| 576 x 768 x 1024 | 1 | 44.6 | 0.045 | 48 | 20.6 | 28.3 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
