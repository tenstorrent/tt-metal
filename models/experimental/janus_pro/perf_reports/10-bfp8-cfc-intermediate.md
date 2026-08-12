# Stage: 10-bfp8-cfc-intermediate

- source commit: `2c43754521f`
- kernel time (mean of replays 2-10): **15.862 ms**
- change from the previous stage: **-0.433 ms**
- device ops: **271**
- note: c_fc's intermediate narrowed to bfloat8_b; c_proj is its only consumer.

## What this change was

**bfloat8_b c_fc intermediate.** Same rule, applied to the MLP's inner tensor. `c_proj` is its only consumer.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 89.88 | -4.65 | 8.899 | 56.2 |
| SDPAOperation | 24 | +0 | 103.40 | -0.19 | 2.482 | 15.7 |
| LayerNormDeviceOperation | 49 | +0 | 32.44 | -0.11 | 1.590 | 10.0 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 50.94 | +0.01 | 1.222 | 7.7 |
| BinaryNgDeviceOperation | 50 | +0 | 21.68 | +0.06 | 1.084 | 6.8 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.97 | +0.02 | 0.431 | 2.7 |
| UnaryDeviceOperation | 1 | +0 | 123.83 | +0.36 | 0.124 | 0.8 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 122.8 | -11.2 | 2.948 | 64 | 29.9 | 15.2 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 96.1 | -6.8 | 2.306 | 48 | 51.0 | 19.4 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 83.1 | -0.3 | 1.994 | 48 | 44.2 | 25.5 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 40.3 | +0.0 | 0.968 | 48 | 30.4 | 24.3 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 459.4 | -24.3 | 0.459 | 48 | 42.6 | 32.5 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 180.0 | +6.0 | 0.180 | 48 | 27.2 | 27.6 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 42.6 | -2.6 | 0.043 | 48 | 21.5 | 29.6 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
