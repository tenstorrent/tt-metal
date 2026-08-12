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

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 83.73 | -2.22 | 8.290 | 55.0 |
| SDPAOperation | 24 | +0 | 102.60 | +0.10 | 2.462 | 16.3 |
| LayerNormDeviceOperation | 49 | +0 | 32.67 | -0.05 | 1.601 | 10.6 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 51.88 | +0.52 | 1.245 | 8.3 |
| BinaryNgDeviceOperation | 50 | +0 | 18.57 | -0.13 | 0.929 | 6.2 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.78 | +0.03 | 0.427 | 2.8 |
| UnaryDeviceOperation | 1 | +0 | 123.53 | -0.99 | 0.124 | 0.8 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 123.0 | -0.0 | 2.951 | 64 | 29.9 | 15.2 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 87.1 | -0.2 | 2.091 | 48 | 56.2 | 19.1 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 82.9 | +0.2 | 1.989 | 48 | 44.3 | 25.5 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 31.8 | -0.0 | 0.763 | 48 | 38.6 | 24.4 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.6 | -182.9 | 0.313 | 48 | 62.7 | 29.1 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 138.8 | -35.2 | 0.139 | 48 | 35.3 | 25.2 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.5 | -1.7 | 0.043 | 48 | 21.1 | 29.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
