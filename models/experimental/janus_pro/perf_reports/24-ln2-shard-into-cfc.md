# Stage: 24-ln2-shard-into-cfc

- source commit: `09c20b5fde7`
- kernel time (mean of replays 2-10): **10.023 ms**
- change from the previous stage: **-0.076 ms**
- device ops: **296**

## What this change was

**ln_2's shard fed to c_fc in place.** Same trick for the MLP, but it required moving `c_fc` to 2D reuse so it could read a shard, and
pinning its `in0_block_w` to 4 to match the shard width in tiles.

2D costs `c_fc` 78.9 → 81.6 us on its own — measured earlier as a standalone change and rejected
as flat. It only pays *here*: the 2D penalty is 0.067 ms across 24 blocks, and the 24 unshards it
removes are 0.143 ms, so it buys back 2.1x what it costs. **320 → 296
ops per replay.**

That is the lesson worth keeping from the whole document: **sharding is chosen for the chain of
ops, not for a single op.** The 8x6 grid is not optimal for the norm in isolation, and 2D is not
optimal for `c_fc` in isolation. Together they are the fastest configuration measured.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 53.85 | +0.84 | 5.331 | 53.2 |
| SDPAOperation | 24 | +0 | 66.98 | -0.75 | 1.607 | 16.0 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.61 | +0.22 | 1.167 | 11.6 |
| LayerNormDeviceOperation | 49 | +0 | 19.09 | +0.29 | 0.935 | 9.3 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.72 | -0.37 | 0.425 | 4.2 |
| ShardedToInterleavedDeviceOperation | 24 | -24 | 9.60 | +1.57 | 0.230 | 2.3 |
| BinaryNgDeviceOperation | 50 | +0 | 3.93 | -0.02 | 0.196 | 2.0 |
| UnaryDeviceOperation | 1 | +0 | 123.71 | -0.10 | 0.124 | 1.2 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 8.31 | +0.08 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | Δ inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc + aligner fc1 | 576 x 1024 x 4096 | 25 | +0 | 81.6 | +2.7 | 2.041 | 48 | 35.2 | 18.6 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | +0 | 55.1 | +0.1 | 1.324 | 48 | 50.2 | 26.4 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | +0 | 48.8 | +0.5 | 1.170 | 48 | 42.6 | 22.4 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | +0 | 18.2 | -0.1 | 0.438 | 48 | 37.9 | 31.2 | LoFi |
| aligner hidden | 576 x 4096 x 4096 | 1 | +0 | 313.9 | +1.1 | 0.314 | 48 | 62.4 | 29.0 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | +0 | 44.2 | +2.1 | 0.044 | 48 | 20.8 | 28.6 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
