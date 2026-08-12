# Stage: 29-encoder-activations-in-l1

- source commit: `81f76bd4f65`
- kernel time (mean of replays 2-10): **9.316 ms**
- change from the previous stage: **-0.085 ms**
- device ops: **293**
- note: patch projection writes ln_1's shard with its bias fused; nlp_concat_heads writes L1

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 52.99 | -0.29 | 5.246 | 56.3 |
| SDPAOperation | 24 | +0 | 65.29 | -0.82 | 1.567 | 16.8 |
| LayerNormDeviceOperation | 49 | +0 | 19.24 | +0.21 | 0.943 | 10.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 35.68 | -0.29 | 0.856 | 9.2 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 13.24 | -0.28 | 0.318 | 3.4 |
| ShardedToInterleavedDeviceOperation | 24 | +0 | 9.66 | +0.05 | 0.232 | 2.5 |
| BinaryNgDeviceOperation | 49 | -1 | 3.06 | -0.86 | 0.150 | 1.6 |
| InterleavedToShardedDeviceOperation | 0 | -1 | — | gone | 0.000 | 0.0 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 72.7 | +0.1 | 1.744 | 48 | 38.1 | 20.0 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.6 | -0.2 | 1.310 | 48 | 50.7 | 26.7 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.6 | +0.4 | 1.167 | 48 | 42.7 | 22.5 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 17.9 | -0.9 | 0.429 | 48 | 38.7 | 20.4 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | 329.7 | +0.3 | 0.330 | 48 | 59.4 | 27.6 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 236.0 | -0.3 | 0.236 | 48 | 20.7 | 13.1 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 29.4 | -14.3 | 0.029 | 48 | 31.2 | 29.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
