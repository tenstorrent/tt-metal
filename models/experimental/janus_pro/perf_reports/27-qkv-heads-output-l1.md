# Stage: 27-qkv-heads-output-l1

- source commit: `a84ca8742c3`
- kernel time (mean of replays 2-10): **9.499 ms**
- change from the previous stage: **-0.342 ms**
- device ops: **295**
- note: nlp_create_qkv_heads writes q/k/v into L1; SDPA is their only consumer

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 53.16 | -0.02 | 5.262 | 55.4 |
| SDPAOperation | 24 | +0 | 65.76 | -1.12 | 1.578 | 16.6 |
| LayerNormDeviceOperation | 49 | +0 | 19.11 | +0.05 | 0.937 | 9.9 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 36.07 | -12.42 | 0.866 | 9.1 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.76 | -0.52 | 0.426 | 4.5 |
| ShardedToInterleavedDeviceOperation | 24 | +0 | 9.63 | -0.05 | 0.231 | 2.4 |
| BinaryNgDeviceOperation | 50 | +0 | 3.96 | +0.01 | 0.198 | 2.1 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.71 | +0.05 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 72.5 | +0.1 | 1.741 | 48 | 38.1 | 20.1 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.9 | -0.1 | 1.317 | 48 | 50.4 | 26.6 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.2 | +0.0 | 1.158 | 48 | 43.0 | 22.6 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.3 | -0.0 | 0.438 | 48 | 37.9 | 31.2 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | 328.4 | -0.7 | 0.328 | 48 | 59.6 | 27.7 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 236.0 | -0.4 | 0.236 | 48 | 20.8 | 13.1 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 44.1 | +0.2 | 0.044 | 48 | 20.8 | 28.6 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
