# Stage: 16-explicit-2d-configs

- source commit: `45b59edfb69`
- kernel time (mean of replays 2-10): **13.481 ms**
- change from the previous stage: **-0.350 ms**
- device ops: **367**
- note: Explicit 2D program configs with in0_block_w swept per shape.

## What this change was

**Explicit 2D configs, in0_block_w per shape.** `in0_block_w` sets how many K-tiles a matmul stages per block. Sweeping each shape in-model found
`qkv` 4, `wo` 8, `c_proj` 16 — and `c_proj` alone gave **−10.6%**.

**Trap: a partial sweep looks conclusive.** ttnn's derived value *is* optimal for `qkv`, so
sweeping only that shape supports "leave it to ttnn". `c_proj` disproves it. Sweep every shape,
or claim nothing.

**Sweep in the model, never standalone.** An isolated bench of the same three shapes disagreed
with the in-model result on two of them.

These values were re-swept in both directions after change 22 changed the fidelity, on the theory
that the compute/read balance had moved. It had not: larger (8/16/32) measured +1.1%, smaller
(2/4/8) +3.0%. The values are a genuine two-sided optimum.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 69.45 | -3.26 | 6.875 | 50.9 |
| SDPAOperation | 24 | +0 | 95.10 | +0.59 | 2.282 | 16.9 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 50.91 | -0.10 | 1.222 | 9.0 |
| LayerNormDeviceOperation | 49 | +0 | 20.18 | +0.25 | 0.989 | 7.3 |
| BinaryNgDeviceOperation | 50 | +0 | 18.51 | -0.12 | 0.925 | 6.9 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.70 | +0.07 | 0.425 | 3.1 |
| InterleavedToShardedDeviceOperation | 48 | +0 | 7.40 | -0.61 | 0.355 | 2.6 |
| ShardedToInterleavedDeviceOperation | 48 | +0 | 6.32 | +0.02 | 0.304 | 2.2 |
| UnaryDeviceOperation | 1 | +0 | 125.28 | +2.10 | 0.125 | 0.9 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 80.6 | -0.1 | 1.934 | 64 | 45.6 | 20.6 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 80.5 | -0.2 | 1.933 | 48 | 45.6 | 23.7 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 74.9 | -11.5 | 1.796 | 48 | 65.4 | 22.2 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 29.8 | -1.6 | 0.715 | 48 | 41.1 | 26.0 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 313.9 | +1.1 | 0.314 | 48 | 62.4 | 29.0 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 139.0 | +0.5 | 0.139 | 48 | 35.2 | 25.2 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 44.6 | -0.2 | 0.045 | 48 | 20.6 | 28.3 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
