# Stage: 11-bfp8-branch-outputs

- source commit: `839efe8eb06`
- kernel time (mean of replays 2-10): **15.288 ms**
- change from the previous stage: **-0.574 ms**
- device ops: **271**
- note: wo and c_proj outputs narrowed to bfloat8_b; both feed the residual.

## What this change was

**bfloat8_b wo and c_proj outputs.** Same rule again. These two are the attention and MLP branch contributions to the residual, each
read once by an `add`. This step cost the most accuracy of the three (−0.0096) because both feed
the residual, which is where error accumulates.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 85.96 | 8.510 | 55.7 |
| SDPAOperation | 24 | 102.50 | 2.460 | 16.1 |
| LayerNormDeviceOperation | 49 | 32.72 | 1.603 | 10.5 |
| NlpCreateHeadsDeviceOperation | 24 | 51.36 | 1.233 | 8.1 |
| BinaryNgDeviceOperation | 50 | 18.70 | 0.935 | 6.1 |
| NLPConcatHeadsDeviceOperation | 24 | 17.75 | 0.426 | 2.8 |
| UnaryDeviceOperation | 1 | 124.52 | 0.125 | 0.8 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 125.1 | 3.127 | 64 | 29.8 | 15.7 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 87.3 | 2.096 | 48 | 56.1 | 19.0 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 82.7 | 1.985 | 48 | 44.4 | 25.6 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 31.8 | 0.762 | 48 | 38.6 | 24.4 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 495.5 | 0.495 | 48 | 39.5 | 30.1 | HiFi2 |
| 576 x 768 x 1024 | 1 | 45.2 | 0.045 | 48 | 20.3 | 28.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
