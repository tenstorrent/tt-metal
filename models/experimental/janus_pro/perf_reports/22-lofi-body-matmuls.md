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

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 52.68 | -6.91 | 5.215 | 51.2 |
| SDPAOperation | 24 | +0 | 67.12 | -0.01 | 1.611 | 15.8 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.11 | -0.09 | 1.155 | 11.3 |
| LayerNormDeviceOperation | 49 | +0 | 18.70 | -0.31 | 0.916 | 9.0 |
| ShardedToInterleavedDeviceOperation | 72 | +0 | 7.45 | +0.03 | 0.536 | 5.3 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.73 | -0.05 | 0.426 | 4.2 |
| BinaryNgDeviceOperation | 50 | +0 | 3.95 | +0.03 | 0.198 | 1.9 |
| UnaryDeviceOperation | 1 | +0 | 122.83 | -1.31 | 0.123 | 1.2 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.58 | -0.09 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc + aligner fc1 | 576 x 1024 x 4096 | 25 | 78.8 | -5.2 | 1.970 | 64 | 27.7 | 22.0 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.5 | -11.0 | 1.308 | 48 | 50.7 | 26.7 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 47.5 | -8.7 | 1.141 | 48 | 43.7 | 27.3 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.3 | -3.4 | 0.440 | 48 | 37.8 | 31.1 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.1 | -0.7 | 0.312 | 48 | 62.8 | 29.2 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 44.7 | +1.3 | 0.045 | 48 | 20.6 | 28.3 | HiFi2 |

**The first row is two projections, not one.** `c_fc` runs 24 times per pass and the aligner's
`fc1` once; they share the shape (576 x 1024 x 4096), so this row averages all 25 instances and its
`us each` belongs to neither. Later reports tell them apart by position — the aligner runs after
every block — but that needs the source profile, and this stage's 12 MB CSV is no longer on disk.
Stage 26 shows the two apart: 24 x 72.4 us and 1 x 236.4 us.

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
