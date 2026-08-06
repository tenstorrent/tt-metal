# Stage: 23-ln1-shard-into-qkv

- source commit: `c2548366aa5`
- kernel time (mean of replays 2-10): **10.099 ms**
- change from the previous stage: **-0.101 ms**
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
| MatmulDeviceOperation | 99 | +0 | 53.01 | +0.33 | 5.248 | 51.9 |
| SDPAOperation | 24 | +0 | 67.73 | +0.61 | 1.626 | 16.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.39 | +0.28 | 1.161 | 11.5 |
| LayerNormDeviceOperation | 49 | +0 | 18.80 | +0.10 | 0.921 | 9.1 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 18.09 | +0.36 | 0.434 | 4.3 |
| ShardedToInterleavedDeviceOperation | 48 | -24 | 8.03 | +0.58 | 0.385 | 3.8 |
| BinaryNgDeviceOperation | 50 | +0 | 3.95 | +0.00 | 0.197 | 2.0 |
| UnaryDeviceOperation | 1 | +0 | 123.81 | +0.98 | 0.124 | 1.2 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 8.23 | +0.65 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | Δ inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc + aligner fc1 | 576 x 1024 x 4096 | 25 | +0 | 78.9 | +0.1 | 1.972 | 64 | 27.6 | 21.9 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | +0 | 55.0 | +0.5 | 1.321 | 48 | 50.3 | 26.5 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | +0 | 48.3 | +0.8 | 1.160 | 48 | 42.9 | 22.6 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | +0 | 18.3 | +0.0 | 0.440 | 48 | 37.7 | 31.0 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | +0 | 312.8 | +0.7 | 0.313 | 48 | 62.6 | 29.1 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | +0 | 42.1 | -2.6 | 0.042 | 48 | 21.8 | 30.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
