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

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 59.59 | -0.00 | 5.899 | 54.2 |
| SDPAOperation | 24 | +0 | 67.13 | -20.91 | 1.611 | 14.8 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.20 | -0.02 | 1.157 | 10.6 |
| LayerNormDeviceOperation | 49 | +0 | 19.01 | -0.50 | 0.931 | 8.6 |
| ShardedToInterleavedDeviceOperation | 72 | +0 | 7.42 | +0.06 | 0.535 | 4.9 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.78 | -0.17 | 0.427 | 3.9 |
| BinaryNgDeviceOperation | 50 | +0 | 3.92 | -0.00 | 0.196 | 1.8 |
| UnaryDeviceOperation | 1 | +0 | 124.14 | +0.90 | 0.124 | 1.1 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.67 | +0.43 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 80.7 | -0.1 | 1.937 | 64 | 45.5 | 20.6 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 65.5 | -0.9 | 1.572 | 48 | 74.8 | 22.2 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 56.2 | +0.8 | 1.349 | 48 | 65.4 | 23.1 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 21.7 | +0.1 | 0.521 | 48 | 56.4 | 26.2 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.8 | +0.5 | 0.313 | 48 | 62.6 | 29.1 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 163.8 | +0.2 | 0.164 | 48 | 29.9 | 18.9 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.4 | -0.0 | 0.043 | 48 | 21.2 | 29.1 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
