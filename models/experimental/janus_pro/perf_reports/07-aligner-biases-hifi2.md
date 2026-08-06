# Stage: 07-aligner-biases-hifi2

- source commit: `176d4978e6a`
- kernel time (mean of replays 2-10): **18.944 ms**
- change from the previous stage: **-0.191 ms**
- device ops: **271**
- note: Aligner biases folded in and the aligner dropped to HiFi2.

## What this change was

**Aligner biases fused, aligner to HiFi2.** The aligner had been skipped when the encoder body was converted, so it was still the only module
at HiFi4 with unfused biases. Housekeeping, found by reading rather than by profiler, worth
0.1997 ms.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 115.51 | 11.435 | 60.3 |
| SDPAOperation | 24 | 111.65 | 2.680 | 14.1 |
| LayerNormDeviceOperation | 49 | 32.44 | 1.590 | 8.4 |
| NlpCreateHeadsDeviceOperation | 24 | 63.06 | 1.514 | 8.0 |
| BinaryNgDeviceOperation | 50 | 22.36 | 1.118 | 5.9 |
| NLPConcatHeadsDeviceOperation | 24 | 20.63 | 0.495 | 2.6 |
| UnaryDeviceOperation | 1 | 123.70 | 0.124 | 0.7 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 139.9 | 3.497 | 64 | 26.6 | 24.2 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 134.2 | 3.220 | 48 | 27.4 | 28.5 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 126.5 | 3.036 | 48 | 38.7 | 26.3 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 47.9 | 1.150 | 48 | 25.6 | 32.3 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 488.6 | 0.489 | 48 | 40.1 | 30.6 | HiFi2 |
| 576 x 768 x 1024 | 1 | 44.1 | 0.044 | 48 | 20.8 | 28.6 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
