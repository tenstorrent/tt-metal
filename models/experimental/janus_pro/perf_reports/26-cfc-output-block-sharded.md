# Stage: 26-cfc-output-block-sharded

- source commit: `5ea2daf4c8c`
- kernel time (mean of replays 2-10): **9.841 ms**
- change from the previous stage: **-0.142 ms**
- device ops: **295**

## What this change was

**c_fc's output became an L1 block shard instead of L1 interleaved.** It was the one body matmul
whose output was never sharded -- changes 17 and 18 covered `wo`, `c_proj` and `qkv` -- and it is the
largest of the four.

The reason to look there came from a per-RISC decomposition of every matmul: **BRISC runs for
99-100% of every one of them**, while the FPU inside TRISC is busy only 24-34% of the time. The body
matmuls are limited neither by math nor by DRAM but by the single RISC that reads in1 and writes the
output. c_fc's writeback was the one piece of that still going the expensive way.

Interleaved L1 scatters the output across every core's banks over the NOC. Block-sharded, each core
writes its own rows into its own L1. **72.4 us against 78.0, so -5.7 us on each of 24 instances,
which accounts for the -0.142 ms on its own.** `c_proj` also reads the shard in place -- the shard is
16 tiles wide and its `in0_block_w` is 16 -- for a further 55.0 -> 54.5 us.

**This is not the writer-loop elimination that change 17 got.** BRISC still runs the full op at
99.5%; what changed is that its transactions became local. The kernel-side loop removal
(`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:594,688-691`) applies to DRAM-to-sharded,
which is what 17 did. Here the source was already L1.

A width-sharded version of this same output was measured at **+1.6%** earlier (see DEAD_ENDS.md) and
the geometry explains both results: width-sharding gave each core 18 rows x 2 columns, block-sharding
gives 3 x 16. Sharding pays by the width of the write burst along N, not by the tile count.

Numerically inert -- the mlp, block and transformer PCCs are all bit-identical to the stage before.

The unabridged per-op listing this table condenses is in
[OPTIMIZED_OP_LIST.md](OPTIMIZED_OP_LIST.md).

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 53.18 | -1.51 | 5.265 | 53.5 |
| SDPAOperation | 24 | +0 | 66.88 | +0.05 | 1.605 | 16.3 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.49 | -0.04 | 1.164 | 11.8 |
| LayerNormDeviceOperation | 49 | +0 | 19.06 | -0.05 | 0.934 | 9.5 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 18.28 | +0.46 | 0.439 | 4.5 |
| ShardedToInterleavedDeviceOperation | 24 | +0 | 9.68 | +0.11 | 0.232 | 2.4 |
| BinaryNgDeviceOperation | 50 | +0 | 3.95 | +0.02 | 0.197 | 2.0 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.66 | +0.03 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 72.4 | -5.8 | 1.739 | 48 | 38.2 | 20.1 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 55.0 | -0.5 | 1.320 | 48 | 50.3 | 26.5 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.2 | +0.0 | 1.158 | 48 | 43.0 | 22.6 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.3 | +0.0 | 0.439 | 48 | 37.8 | 31.1 | LoFi |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 236.4 | +0.3 | 0.236 | 48 | 20.7 | 13.1 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 329.1 | +1.6 | 0.329 | 48 | 59.5 | 27.7 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.9 | -0.5 | 0.044 | 48 | 20.9 | 28.8 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
