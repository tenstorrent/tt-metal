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

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 136.78 | -9.63 | 13.541 | 57.5 |
| BinaryNgDeviceOperation | 124 | +0 | 28.83 | +0.17 | 3.575 | 15.2 |
| SDPAOperation | 24 | +0 | 111.56 | +1.13 | 2.677 | 11.4 |
| LayerNormDeviceOperation | 49 | +0 | 32.45 | +0.09 | 1.590 | 6.8 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 63.71 | -0.67 | 1.529 | 6.5 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 20.89 | +0.18 | 0.501 | 2.1 |
| UnaryDeviceOperation | 1 | +0 | 122.68 | -0.10 | 0.123 | 0.5 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 234.7 | -38.9 | 5.634 | 64 | 15.6 | 14.2 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 128.2 | -0.1 | 3.077 | 48 | 28.6 | 29.8 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 125.2 | +0.1 | 3.004 | 48 | 39.1 | 26.5 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 46.3 | -0.9 | 1.112 | 48 | 26.4 | 33.4 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 179.1 | -2.3 | 0.179 | 48 | 54.7 | 27.7 | HiFi4 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 490.5 | +1.2 | 0.490 | 48 | 79.9 | 30.4 | HiFi4 |
| patch embed | 576 x 768 x 1024 | 1 | 44.2 | -0.4 | 0.044 | 48 | 20.8 | 28.6 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
