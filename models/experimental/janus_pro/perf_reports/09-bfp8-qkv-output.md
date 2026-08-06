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

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 94.53 | 9.358 | 57.4 |
| SDPAOperation | 24 | 103.59 | 2.486 | 15.3 |
| LayerNormDeviceOperation | 49 | 32.55 | 1.595 | 9.8 |
| NlpCreateHeadsDeviceOperation | 24 | 50.93 | 1.222 | 7.5 |
| BinaryNgDeviceOperation | 50 | 21.62 | 1.081 | 6.6 |
| NLPConcatHeadsDeviceOperation | 24 | 17.95 | 0.431 | 2.6 |
| UnaryDeviceOperation | 1 | 123.47 | 0.123 | 0.8 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 135.6 | 3.390 | 64 | 27.4 | 14.5 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 102.9 | 2.470 | 48 | 47.6 | 18.1 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 83.4 | 2.002 | 48 | 44.0 | 25.4 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 40.3 | 0.967 | 48 | 30.4 | 24.3 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 483.7 | 0.484 | 48 | 40.5 | 30.9 | HiFi2 |
| 576 x 768 x 1024 | 1 | 45.2 | 0.045 | 48 | 20.3 | 28.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
