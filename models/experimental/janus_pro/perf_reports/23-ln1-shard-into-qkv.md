# Stage: 23-ln1-shard-into-qkv

- source commit: `c2548366aa5`
- kernel time (mean of replays 2-10): **10.089 ms**
- change from the previous stage: **-0.109 ms**
- device ops: **320**

## What this change was

**ln_1's shard fed to qkv in place.** Not a knob — the only structural op elimination left. `_size_shard(576, 1024)` puts the norm on an
8x6 grid with 3x4-tile blocks. That is **exactly** the in0 shard a 2D mcast matmul on the same
grid wants, namely `per_core_M x K/grid_x = 3 x 4`, and qkv's `in0_block_w` was already 4. So the
unshard between them was pure overhead.

**344 → 320 ops per replay.** The tensor stays bfloat16: under LoFi the mantissa is truncated to
the same 5 bits either way, so narrowing first only loses bits. Both `test_vision_transformer` and
e2e PCC went *up*.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 52.98 | +0.37 | 5.245 | 52.0 |
| SDPAOperation | 24 | +0 | 67.30 | +0.04 | 1.615 | 16.0 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.47 | +0.16 | 1.163 | 11.5 |
| LayerNormDeviceOperation | 49 | +0 | 18.82 | +0.02 | 0.922 | 9.1 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.88 | -0.16 | 0.429 | 4.3 |
| ShardedToInterleavedDeviceOperation | 48 | -24 | 8.04 | +0.58 | 0.386 | 3.8 |
| BinaryNgDeviceOperation | 50 | +0 | 3.92 | -0.04 | 0.196 | 1.9 |
| UnaryDeviceOperation | 1 | +0 | 124.08 | +0.69 | 0.124 | 1.2 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.77 | +0.19 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 75.5 | +0.3 | 1.812 | 64 | 27.5 | 22.0 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 54.7 | +0.2 | 1.313 | 48 | 50.6 | 26.6 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.3 | +1.0 | 1.158 | 48 | 43.0 | 22.6 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.4 | +0.1 | 0.441 | 48 | 37.7 | 31.0 | LoFi |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 163.9 | +0.5 | 0.164 | 48 | 29.9 | 18.9 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.6 | -1.0 | 0.313 | 48 | 62.7 | 29.1 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 44.7 | +0.3 | 0.045 | 48 | 20.5 | 28.2 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
