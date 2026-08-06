# Stage: 05-qkv-bias-fused

- source commit: `76e533f24e9`
- kernel time (mean of replays 2-10): **20.037 ms**
- change from the previous stage: **-1.178 ms**
- device ops: **321**
- note: qkv bias folded into its matmul.

## What this change was

**qkv bias fused into its matmul.** Under trace each removed elementwise op is ~29 us of pure kernel time, almost independent of how
little arithmetic it does. The **qkv bias fuses at any device count**, because `wqkv` and `bqkv`
shard on the same axis with nothing reducing between them, so the per-device fused result is the
plain sum.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 114.95 | 11.380 | 56.8 |
| SDPAOperation | 24 | 110.10 | 2.642 | 13.2 |
| BinaryNgDeviceOperation | 100 | 22.77 | 2.277 | 11.4 |
| LayerNormDeviceOperation | 49 | 32.34 | 1.585 | 7.9 |
| NlpCreateHeadsDeviceOperation | 24 | 63.66 | 1.528 | 7.6 |
| NLPConcatHeadsDeviceOperation | 24 | 20.58 | 0.494 | 2.5 |
| UnaryDeviceOperation | 1 | 123.29 | 0.123 | 0.6 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 24 | 138.6 | 3.327 | 64 | 26.5 | 24.0 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 134.6 | 3.230 | 48 | 27.3 | 28.4 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 124.9 | 2.998 | 48 | 39.2 | 26.6 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 46.3 | 1.111 | 48 | 26.5 | 33.4 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 490.6 | 0.491 | 48 | 79.9 | 30.4 | HiFi4 |
| 576 x 1024 x 4096 | 1 | 180.9 | 0.181 | 48 | 54.1 | 27.4 | HiFi4 |
| 576 x 768 x 1024 | 1 | 43.4 | 0.043 | 48 | 21.2 | 29.1 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
