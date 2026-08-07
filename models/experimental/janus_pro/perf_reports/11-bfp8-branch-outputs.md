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

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 85.96 | -3.92 | 8.510 | 55.7 |
| SDPAOperation | 24 | +0 | 102.50 | -0.90 | 2.460 | 16.1 |
| LayerNormDeviceOperation | 49 | +0 | 32.72 | +0.28 | 1.603 | 10.5 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 51.36 | +0.42 | 1.233 | 8.1 |
| BinaryNgDeviceOperation | 50 | +0 | 18.70 | -2.98 | 0.935 | 6.1 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.75 | -0.22 | 0.426 | 2.8 |
| UnaryDeviceOperation | 1 | +0 | 124.52 | +0.69 | 0.125 | 0.8 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 123.0 | +0.2 | 2.953 | 64 | 29.9 | 15.2 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 87.3 | -8.8 | 2.096 | 48 | 56.1 | 19.0 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 82.7 | -0.4 | 1.985 | 48 | 44.4 | 25.6 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 31.8 | -8.5 | 0.762 | 48 | 38.6 | 24.4 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 495.5 | +36.1 | 0.495 | 48 | 39.5 | 30.1 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 174.0 | -6.0 | 0.174 | 48 | 28.1 | 28.5 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 45.2 | +2.6 | 0.045 | 48 | 20.3 | 28.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
