# Stage: 28-sdpa-output-l1

- source commit: `6cdad2cf097`
- kernel time (mean of replays 2-10): **9.401 ms**
- change from the previous stage: **-0.098 ms**
- device ops: **295**
- note: SDPA writes into L1; nlp_concat_heads is its only consumer

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 53.28 | +0.12 | 5.275 | 56.0 |
| SDPAOperation | 24 | +0 | 66.11 | +0.35 | 1.587 | 16.8 |
| LayerNormDeviceOperation | 49 | +0 | 19.03 | -0.08 | 0.933 | 9.9 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 35.97 | -0.10 | 0.863 | 9.2 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 13.52 | -4.24 | 0.325 | 3.4 |
| ShardedToInterleavedDeviceOperation | 24 | +0 | 9.61 | -0.02 | 0.231 | 2.5 |
| BinaryNgDeviceOperation | 50 | +0 | 3.92 | -0.04 | 0.196 | 2.1 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.60 | -0.11 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 72.6 | +0.1 | 1.742 | 48 | 38.1 | 20.1 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.8 | -0.1 | 1.316 | 48 | 50.5 | 26.6 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.2 | -0.0 | 1.156 | 48 | 43.1 | 22.7 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.8 | +0.5 | 0.452 | 48 | 36.8 | 30.2 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | 329.4 | +1.0 | 0.329 | 48 | 59.5 | 27.6 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 236.3 | +0.3 | 0.236 | 48 | 20.7 | 13.1 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.7 | -0.4 | 0.044 | 48 | 21.0 | 28.9 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
