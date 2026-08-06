# Stage: 17-wo-cproj-sharded

- source commit: `5a47c877785`
- kernel time (mean of replays 2-10): **12.025 ms**
- change from the previous stage: **-1.456 ms**
- device ops: **320**
- note: wo and c_proj outputs written as L1 block shards.

## What this change was

**wo and c_proj outputs L1 block-sharded.** The mechanism behind changes 17-19. **BRISC hosts the in1 read *and* the output writeback**
(`matmul_multicore_reuse_mcast_2d_program_factory.cpp:735,791,808,822`), so asking for a sharded
output compiles the writer loop out of the kernel entirely
(`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:594,688-691`, set at
`2d_program_factory.cpp:645-648`). The shard spec is derived from `per_core_M`/`per_core_N`, so a
bare `ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG` suffices.

**The gain scales with write-burst width along N, not with tile count:**

| op | per_core_M x per_core_N | tiles/core | op delta |
|---|---|---:|---:|
| qkv | 3 x 12 | 36 | **−31%** |
| wo | 3 x 4 | 12 | −27% |
| c_proj | 3 x 4 | 12 | −12% |
| c_fc (1D) | 18 x 2 | 36 | −4%, does not cover its unshard |

`qkv` and `c_fc` carry the same 36 tiles per core yet differ 31% vs 4%: **the burst geometry, not
the volume, is what matters.**

**Gate the sharded output on supplying the program config.** ttnn's own derivation for the
2432-token shape yields `num_blocks_y=10` against a grid of 8 rows, which a sharded output rejects.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 65.60 | 6.494 | 54.0 |
| SDPAOperation | 24 | 95.02 | 2.280 | 19.0 |
| NlpCreateHeadsDeviceOperation | 24 | 51.31 | 1.231 | 10.2 |
| LayerNormDeviceOperation | 49 | 19.40 | 0.951 | 7.9 |
| NLPConcatHeadsDeviceOperation | 24 | 18.14 | 0.435 | 3.6 |
| ShardedToInterleavedDeviceOperation | 48 | 6.39 | 0.307 | 2.6 |
| BinaryNgDeviceOperation | 50 | 3.93 | 0.197 | 1.6 |
| UnaryDeviceOperation | 1 | 123.23 | 0.123 | 1.0 |
| InterleavedToShardedDeviceOperation | 1 | 7.72 | 0.008 | 0.1 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 84.0 | 2.099 | 64 | 44.9 | 20.5 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 80.5 | 1.933 | 48 | 45.6 | 23.7 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 66.1 | 1.587 | 48 | 74.1 | 22.0 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 21.6 | 0.519 | 48 | 56.6 | 26.3 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 313.2 | 0.313 | 48 | 62.6 | 29.1 | HiFi2 |
| 576 x 768 x 1024 | 1 | 43.2 | 0.043 | 48 | 21.3 | 29.3 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
