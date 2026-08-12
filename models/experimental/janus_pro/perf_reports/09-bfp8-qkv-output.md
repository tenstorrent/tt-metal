# Stage: 09-bfp8-qkv-output

- source commit: `9a559f35bed`
- kernel time (mean of replays 2-10): **16.295 ms**
- change from the previous stage: **-1.276 ms**
- device ops: **271**
- note: Fused qkv output narrowed to bfloat8_b; the format propagates through SDPA into wo.

## What this change was

**bfloat8_b fused qkv output.** The same lever one tensor further along. The criterion for whether a tensor may be narrowed is
**read-once versus accumulated**: a tensor with a single consumer narrows for free, an accumulator
does not. **Size is irrelevant.**

The qkv output propagates its format through `nlp_create_qkv_heads`, SDPA and `nlp_concat_heads`
into the `wo` matmul, so one dtype change moved four ops.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 94.53 | -6.70 | 9.358 | 57.4 |
| SDPAOperation | 24 | +0 | 103.59 | -8.70 | 2.486 | 15.3 |
| LayerNormDeviceOperation | 49 | +0 | 32.55 | +0.00 | 1.595 | 9.8 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 50.93 | -12.04 | 1.222 | 7.5 |
| BinaryNgDeviceOperation | 50 | +0 | 21.62 | -0.72 | 1.081 | 6.6 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.95 | -2.73 | 0.431 | 2.6 |
| UnaryDeviceOperation | 1 | +0 | 123.47 | -0.84 | 0.123 | 0.8 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 134.0 | -0.0 | 3.216 | 64 | 27.4 | 13.9 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 102.9 | +0.0 | 2.470 | 48 | 47.6 | 18.1 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 83.4 | -25.5 | 2.002 | 48 | 44.0 | 25.4 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 40.3 | -2.8 | 0.967 | 48 | 30.4 | 24.3 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 483.7 | +13.5 | 0.484 | 48 | 40.5 | 30.9 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 174.0 | -1.8 | 0.174 | 48 | 28.1 | 28.5 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 45.2 | +2.0 | 0.045 | 48 | 20.3 | 28.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
