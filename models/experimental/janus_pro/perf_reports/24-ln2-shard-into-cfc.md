# Stage: 24-ln2-shard-into-cfc

- source commit: `09c20b5fde7`
- kernel time (mean of replays 2-10): **10.016 ms**
- change from the previous stage: **-0.073 ms**
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
| MatmulDeviceOperation | 99 | +0 | 53.74 | +0.76 | 5.320 | 53.1 |
| SDPAOperation | 24 | +0 | 67.10 | -0.20 | 1.610 | 16.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.35 | -0.12 | 1.160 | 11.6 |
| LayerNormDeviceOperation | 49 | +0 | 19.01 | +0.19 | 0.931 | 9.3 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 18.26 | +0.38 | 0.438 | 4.4 |
| ShardedToInterleavedDeviceOperation | 24 | -24 | 9.55 | +1.51 | 0.229 | 2.3 |
| BinaryNgDeviceOperation | 50 | +0 | 3.94 | +0.02 | 0.197 | 2.0 |
| UnaryDeviceOperation | 1 | +0 | 123.02 | -1.06 | 0.123 | 1.2 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.47 | -0.30 | 0.007 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 78.1 | +2.6 | 1.874 | 48 | 35.4 | 18.7 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 55.2 | +0.5 | 1.326 | 48 | 50.1 | 26.4 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.4 | +0.1 | 1.162 | 48 | 42.9 | 22.6 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.3 | -0.1 | 0.439 | 48 | 37.8 | 31.1 | LoFi |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 164.7 | +0.8 | 0.165 | 48 | 29.7 | 18.8 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.0 | -0.6 | 0.312 | 48 | 62.8 | 29.2 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.5 | -1.2 | 0.043 | 48 | 21.1 | 29.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
