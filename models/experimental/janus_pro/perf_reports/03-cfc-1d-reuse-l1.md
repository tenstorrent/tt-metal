# Stage: 03-cfc-1d-reuse-l1

- source commit: `8dfc3cf9198`
- kernel time (mean of replays 2-10): **23.536 ms**
- change from the previous stage: **-0.909 ms**
- device ops: **345**
- note: c_fc as 1D reuse with its output kept in L1.

## What this change was

**c_fc as 1D reuse, output in L1.** N is 4096 against M 576, so the matmul is narrow. 1D multicasts in0 and slices N, putting all 64
cores to work where the derived 2D grid reached only 48. Keeping the intermediate in L1 saves
`c_proj` a DRAM round trip. **272 → 233 us per op.**

The same switch on `qkv` was 33% *slower* — its 96 N-tiles cannot spread past 48 cores. Same
lever, opposite result, decided by shape.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 136.78 | 13.541 | 57.5 |
| BinaryNgDeviceOperation | 124 | 28.83 | 3.575 | 15.2 |
| SDPAOperation | 24 | 111.56 | 2.677 | 11.4 |
| LayerNormDeviceOperation | 49 | 32.45 | 1.590 | 6.8 |
| NlpCreateHeadsDeviceOperation | 24 | 63.71 | 1.529 | 6.5 |
| NLPConcatHeadsDeviceOperation | 24 | 20.89 | 0.501 | 2.1 |
| UnaryDeviceOperation | 1 | 122.68 | 0.123 | 0.5 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 232.5 | 5.813 | 64 | 17.2 | 14.7 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 128.2 | 3.077 | 48 | 28.6 | 29.8 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 125.2 | 3.004 | 48 | 39.1 | 26.5 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 46.3 | 1.112 | 48 | 26.4 | 33.4 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 490.5 | 0.490 | 48 | 79.9 | 30.4 | HiFi4 |
| 576 x 768 x 1024 | 1 | 44.2 | 0.044 | 48 | 20.8 | 28.6 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
