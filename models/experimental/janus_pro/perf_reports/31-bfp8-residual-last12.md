# Stage: 31-bfp8-residual-last12

- source commit: `8b6e2547bd9`
- kernel time (mean of replays 2-10): **9.230 ms**
- change from the previous stage: **-0.064 ms**
- device ops: **293**
- note: bfloat8_b residual on encoder blocks 12-23

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 52.37 | -0.47 | 5.184 | 56.2 |
| SDPAOperation | 24 | +0 | 65.63 | -0.32 | 1.575 | 17.1 |
| LayerNormDeviceOperation | 49 | +0 | 18.94 | -0.22 | 0.928 | 10.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 35.42 | -0.11 | 0.850 | 9.2 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 13.36 | +0.14 | 0.321 | 3.5 |
| ShardedToInterleavedDeviceOperation | 24 | +0 | 9.61 | +0.04 | 0.231 | 2.5 |
| BinaryNgDeviceOperation | 49 | +0 | 2.91 | -0.15 | 0.143 | 1.5 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 71.4 | -1.0 | 1.714 | 48 | 38.8 | 20.4 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.8 | -0.1 | 1.316 | 48 | 50.5 | 26.6 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.2 | -0.6 | 1.157 | 48 | 43.1 | 22.7 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 17.9 | -0.1 | 0.429 | 48 | 38.7 | 20.4 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.6 | -2.3 | 0.313 | 48 | 62.7 | 26.5 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 227.8 | -3.1 | 0.228 | 48 | 21.5 | 10.0 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 28.9 | -0.6 | 0.029 | 48 | 31.7 | 29.5 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
