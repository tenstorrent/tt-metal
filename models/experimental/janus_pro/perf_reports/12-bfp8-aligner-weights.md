# Stage: 12-bfp8-aligner-weights

- source commit: `0b4a907677e`
- kernel time (mean of replays 2-10): **15.073 ms**
- change from the previous stage: **-0.215 ms**
- device ops: **271**
- note: Aligner weights narrowed to bfloat8_b.

## What this change was

**bfloat8_b aligner weights.** The aligner's **576x4096x4096** layer is the single most expensive matmul instance in the tower.
Left in bfloat16 when the body was converted; dropped **490.9 → 312.9 us**.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 83.73 | 8.290 | 55.0 |
| SDPAOperation | 24 | 102.60 | 2.462 | 16.3 |
| LayerNormDeviceOperation | 49 | 32.67 | 1.601 | 10.6 |
| NlpCreateHeadsDeviceOperation | 24 | 51.88 | 1.245 | 8.3 |
| BinaryNgDeviceOperation | 50 | 18.57 | 0.929 | 6.2 |
| NLPConcatHeadsDeviceOperation | 24 | 17.78 | 0.427 | 2.8 |
| UnaryDeviceOperation | 1 | 123.53 | 0.124 | 0.8 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 123.6 | 3.090 | 64 | 30.1 | 15.6 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 87.1 | 2.091 | 48 | 56.2 | 19.1 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 82.9 | 1.989 | 48 | 44.3 | 25.5 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 31.8 | 0.763 | 48 | 38.6 | 24.4 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 312.6 | 0.313 | 48 | 62.7 | 29.1 | HiFi2 |
| 576 x 768 x 1024 | 1 | 43.5 | 0.043 | 48 | 21.1 | 29.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
