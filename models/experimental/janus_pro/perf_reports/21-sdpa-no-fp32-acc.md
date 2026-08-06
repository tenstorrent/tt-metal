# Stage: 21-sdpa-no-fp32-acc

- source commit: `a709b232923`
- kernel time (mean of replays 2-10): **10.891 ms**
- change from the previous stage: **-0.509 ms**
- device ops: **344**
- note: fp32 destination accumulation turned off on SDPA.

## What this change was

**fp32 dest accumulation off on SDPA.** `fp32_dest_acc_en` halves the DST register budget (`compute_kernel_config.cpp:152-161`). Turning
it off took SDPA 87.76 → 67.42 us, **−23.2%**, and tower PCC *rose* to 0.974919 — above even what
it was before SDPA was touched at all.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 59.59 | 5.899 | 54.2 |
| SDPAOperation | 24 | 67.13 | 1.611 | 14.8 |
| NlpCreateHeadsDeviceOperation | 24 | 48.20 | 1.157 | 10.6 |
| LayerNormDeviceOperation | 49 | 19.01 | 0.931 | 8.6 |
| ShardedToInterleavedDeviceOperation | 72 | 7.42 | 0.535 | 4.9 |
| NLPConcatHeadsDeviceOperation | 24 | 17.78 | 0.427 | 3.9 |
| BinaryNgDeviceOperation | 50 | 3.92 | 0.196 | 1.8 |
| UnaryDeviceOperation | 1 | 124.14 | 0.124 | 1.1 |
| InterleavedToShardedDeviceOperation | 1 | 7.67 | 0.008 | 0.1 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 84.0 | 2.101 | 64 | 44.9 | 20.5 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 65.5 | 1.572 | 48 | 74.8 | 22.2 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 56.2 | 1.349 | 48 | 65.4 | 23.1 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 21.7 | 0.521 | 48 | 56.4 | 26.2 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 312.8 | 0.313 | 48 | 62.6 | 29.1 | HiFi2 |
| 576 x 768 x 1024 | 1 | 43.4 | 0.043 | 48 | 21.2 | 29.1 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
