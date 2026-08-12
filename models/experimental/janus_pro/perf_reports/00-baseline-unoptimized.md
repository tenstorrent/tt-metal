# Stage: 00-baseline-unoptimized

- source commit: `f7f7d7cd87f`
- kernel time (mean of replays 2-10): **29.501 ms**
- device ops: **393**
- note: Baseline compute path with today's trace plumbing; same harness as every later stage (warm run, trace on, 10 replays).

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 128.02 | 12.674 | 43.1 |
| BinaryNgDeviceOperation | 148 | 38.70 | 5.728 | 19.5 |
| SDPAOperation | 24 | 181.89 | 4.365 | 14.8 |
| UnaryDeviceOperation | 25 | 122.34 | 3.058 | 10.4 |
| LayerNormDeviceOperation | 49 | 32.21 | 1.578 | 5.4 |
| NlpCreateHeadsDeviceOperation | 24 | 63.80 | 1.531 | 5.2 |
| NLPConcatHeadsDeviceOperation | 24 | 20.82 | 0.500 | 1.7 |

## Matmul instances by shape

| layer | shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 179.2 | 4.301 | 48 | 54.7 | 27.7 | HiFi4 |
| attn qkv | 576 x 1024 x 3072 | 24 | 138.1 | 3.315 | 48 | 53.2 | 27.7 | HiFi4 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 130.5 | 3.133 | 48 | 75.0 | 38.0 | HiFi4 |
| attn wo | 576 x 1024 x 1024 | 24 | 50.5 | 1.212 | 48 | 48.5 | 30.6 | HiFi4 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 490.8 | 0.491 | 48 | 79.8 | 30.4 | HiFi4 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 177.4 | 0.177 | 48 | 55.2 | 28.0 | HiFi4 |
| patch embed | 576 x 768 x 1024 | 1 | 44.9 | 0.045 | 48 | 20.5 | 28.1 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
