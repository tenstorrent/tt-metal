# Stage: 22-lofi-body-matmuls

- source commit: `744b677f553`
- kernel time (mean of replays 2-10): **10.198 ms**
- change from the previous stage: **-0.693 ms**
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

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 52.61 | -6.98 | 5.208 | 51.1 |
| SDPAOperation | 24 | +0 | 67.26 | +0.13 | 1.614 | 15.8 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.31 | +0.11 | 1.159 | 11.4 |
| LayerNormDeviceOperation | 49 | +0 | 18.80 | -0.21 | 0.921 | 9.0 |
| ShardedToInterleavedDeviceOperation | 72 | +0 | 7.46 | +0.04 | 0.537 | 5.3 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 18.04 | +0.26 | 0.433 | 4.2 |
| BinaryNgDeviceOperation | 50 | +0 | 3.96 | +0.04 | 0.198 | 1.9 |
| UnaryDeviceOperation | 1 | +0 | 123.39 | -0.75 | 0.123 | 1.2 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.58 | -0.09 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 75.2 | -5.5 | 1.805 | 64 | 27.6 | 22.1 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.5 | -11.0 | 1.308 | 48 | 50.8 | 26.7 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 47.3 | -8.9 | 1.134 | 48 | 43.9 | 27.5 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.3 | -3.4 | 0.439 | 48 | 37.8 | 31.1 | LoFi |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 163.4 | -0.4 | 0.163 | 48 | 30.0 | 18.9 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 313.6 | +0.8 | 0.314 | 48 | 62.5 | 29.0 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 44.4 | +1.0 | 0.044 | 48 | 20.7 | 28.4 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
