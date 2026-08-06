# Stage: 18-qkv-output-sharded

- source commit: `4b6b23530ba`
- kernel time (mean of replays 2-10): **11.737 ms**
- change from the previous stage: **-0.288 ms**
- device ops: **344**
- note: qkv output written as an L1 block shard too.

## What this change was

**qkv output L1 block-sharded.** Same mechanism, applied to the widest burst in the tower (3 x 12 tiles per core).

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 59.55 | 5.896 | 50.2 |
| SDPAOperation | 24 | 94.99 | 2.280 | 19.4 |
| NlpCreateHeadsDeviceOperation | 24 | 50.87 | 1.221 | 10.4 |
| LayerNormDeviceOperation | 49 | 19.52 | 0.957 | 8.1 |
| ShardedToInterleavedDeviceOperation | 72 | 8.83 | 0.636 | 5.4 |
| NLPConcatHeadsDeviceOperation | 24 | 17.90 | 0.430 | 3.7 |
| BinaryNgDeviceOperation | 50 | 3.87 | 0.193 | 1.6 |
| UnaryDeviceOperation | 1 | 123.61 | 0.124 | 1.1 |
| InterleavedToShardedDeviceOperation | 1 | 7.62 | 0.008 | 0.1 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 83.9 | 2.098 | 64 | 44.9 | 20.5 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 66.3 | 1.591 | 48 | 73.9 | 22.0 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 55.4 | 1.330 | 48 | 66.3 | 23.4 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 21.7 | 0.521 | 48 | 56.4 | 26.2 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 312.5 | 0.313 | 48 | 62.7 | 29.1 | HiFi2 |
| 576 x 768 x 1024 | 1 | 43.7 | 0.044 | 48 | 21.0 | 28.9 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
