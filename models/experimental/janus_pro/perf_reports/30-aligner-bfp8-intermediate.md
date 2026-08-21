# Stage: 30-aligner-bfp8-intermediate

- source commit: `8daaa3f6dd7`
- kernel time (mean of replays 2-10): **9.294 ms**
- change from the previous stage: **-0.022 ms**
- device ops: **293**
- note: aligner intermediate projections emit bfloat8_b

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 52.84 | -0.15 | 5.231 | 56.2 |
| SDPAOperation | 24 | +0 | 65.95 | +0.66 | 1.583 | 17.0 |
| LayerNormDeviceOperation | 49 | +0 | 19.16 | -0.08 | 0.939 | 10.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 35.53 | -0.15 | 0.853 | 9.2 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 13.22 | -0.02 | 0.317 | 3.4 |
| ShardedToInterleavedDeviceOperation | 24 | +0 | 9.57 | -0.09 | 0.230 | 2.5 |
| BinaryNgDeviceOperation | 49 | +0 | 3.06 | +0.00 | 0.150 | 1.6 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 72.4 | -0.3 | 1.737 | 48 | 38.2 | 20.1 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.9 | +0.3 | 1.317 | 48 | 50.5 | 26.6 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.8 | +0.2 | 1.171 | 48 | 42.6 | 22.4 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.0 | +0.1 | 0.431 | 48 | 38.5 | 20.3 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | 314.9 | -14.8 | 0.315 | 48 | 62.2 | 26.3 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 230.9 | -5.1 | 0.231 | 48 | 21.2 | 9.9 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 29.5 | +0.1 | 0.030 | 48 | 31.1 | 28.9 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
