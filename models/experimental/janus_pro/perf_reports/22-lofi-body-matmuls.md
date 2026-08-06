# Stage: 22-lofi-body-matmuls

- source commit: `744b677f553`
- kernel time (mean of replays 2-10): **10.200 ms**
- change from the previous stage: **-0.691 ms**
- device ops: **344**

## What this change was

**LoFi on the body matmuls.** The largest single win among the fidelity changes. `qkv`, `wo`, `c_fc` and `c_proj` all take
**bfloat8_b on both sides**. bfloat8_b carries a 7-bit mantissa; LoFi's single pass covers about 5
of them, so HiFi2's second pass was reading bits the operands do not have.

Peak per core goes 2.056 → 3.639 TFLOPs, a factor of 1.77. Measured **−15 to −17% per
projection**, matmul block 5.896 → 5.215 ms — not the 44% the peak ratio implies, because these
matmuls are only partly math-bound.

The aligner was deliberately **left at HiFi2**: its activations are bfloat16, so there LoFi is a
real precision loss rather than a free one, and it feeds the language model directly.

A free-looking follow-on did not pay: turning fp32 acc off doubles the real DST bound from 4 to 8
tiles, so `get_out_subblock_w`'s hardcoded 4 leaves half unused. Taking it (qkv 4→6, c_fc h 2→3)
measured flat and bit-identical. DST was not the constraint.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 52.68 | 5.215 | 51.2 |
| SDPAOperation | 24 | 67.12 | 1.611 | 15.8 |
| NlpCreateHeadsDeviceOperation | 24 | 48.11 | 1.155 | 11.3 |
| LayerNormDeviceOperation | 49 | 18.70 | 0.916 | 9.0 |
| ShardedToInterleavedDeviceOperation | 72 | 7.45 | 0.536 | 5.3 |
| NLPConcatHeadsDeviceOperation | 24 | 17.73 | 0.426 | 4.2 |
| BinaryNgDeviceOperation | 50 | 3.95 | 0.198 | 1.9 |
| UnaryDeviceOperation | 1 | 122.83 | 0.123 | 1.2 |
| InterleavedToShardedDeviceOperation | 1 | 7.58 | 0.008 | 0.1 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 78.8 | 1.970 | 64 | 27.7 | 22.0 | LoFi |
| 576 x 4096 x 1024 | 24 | 54.5 | 1.308 | 48 | 50.7 | 26.7 | LoFi |
| 576 x 1024 x 3072 | 24 | 47.5 | 1.141 | 48 | 43.7 | 27.3 | LoFi |
| 576 x 1024 x 1024 | 24 | 18.3 | 0.440 | 48 | 37.8 | 31.1 | LoFi |
| 576 x 4096 x 4096 | 1 | 312.1 | 0.312 | 48 | 62.8 | 29.2 | HiFi2 |
| 576 x 768 x 1024 | 1 | 44.7 | 0.045 | 48 | 20.6 | 28.3 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
