# Complete operator profile

Full 64-layer model plus split device sampler, real weights and synthetic populated caches; three restored-state trace replays, not natural-prompt accuracy or unprofiled serving throughput

Context 32768, batch 16, 33 operation types.
Unprofiled restored-cache step 50.087 ms; profiled 52.306 ms.
Whole-step profiler overhead 4.43%; this is not HTTP/natural-prompt throughput.

Each rank's firmware span is partitioned by op family with overlap and gaps explicit. Use each replay's longest-rank timeline; do not sum rank times or family medians. Firmware includes waits. Profile overhead is measured for the full step, not per operation.

| Operation | Calls, range | Median kernel sum, ms |
|---|---:|---:|
| MatmulDeviceOperation | 260-260 | 18.9871 |
| SdpaDecodeDeviceOperation | 16-16 | 12.6267 |
| GenericOpDeviceOperation | 192-192 | 9.1845 |
| AllReduceAsyncDeviceOperation | 128-128 | 2.6468 |
| BinaryNgDeviceOperation | 316-316 | 1.2623 |
| LayerNormDeviceOperation | 161-161 | 1.0884 |
| ReshapeViewDeviceOperation | 130-130 | 0.9594 |
| SliceDeviceOperation | 497-497 | 0.9458 |
| ReshardDeviceOperation | 193-193 | 0.5589 |
| UnaryDeviceOperation | 97-97 | 0.3680 |
| TopkLargeIndicesDeviceOperation | 1-1 | 0.2723 |
| ConcatDeviceOperation | 33-33 | 0.2389 |
| ShardedToInterleavedDeviceOperation | 164-164 | 0.2241 |
| TransposeDeviceOperation | 64-64 | 0.2065 |
| InterleavedToShardedDeviceOperation | 146-146 | 0.2017 |
| RotaryEmbeddingDeviceOperation | 32-32 | 0.1370 |
| FillPadDeviceOperation | 17-17 | 0.1084 |
| TypecastDeviceOperation | 52-52 | 0.0894 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 16-16 | 0.0736 |
| PagedFusedUpdateCacheDeviceOperation | 16-16 | 0.0707 |
| TopkRoutePrepDeviceOperation | 1-1 | 0.0243 |
| SamplingDeviceOperation | 1-1 | 0.0215 |
| TopkRouteFinishDeviceOperation | 1-1 | 0.0205 |
| AllGatherAsyncDeviceOperation | 1-1 | 0.0192 |
| AllGatherDeviceOperation | 2-2 | 0.0189 |
| ManualSeedDeviceOperation | 1-1 | 0.0186 |
| TilizeWithValPaddingDeviceOperation | 3-3 | 0.0155 |
| EmbeddingsDeviceOperation | 3-3 | 0.0070 |
| CopyDeviceOperation | 1-1 | 0.0062 |
| UntilizeDeviceOperation | 1-1 | 0.0043 |
| ReduceDeviceOperation | 2-2 | 0.0042 |
| PlusOneDeviceOperation | 4-4 | 0.0031 |
| IndexedFillDeviceOperation | 1-1 | 0.0029 |

## RISC time including waits

These durations are not active compute or physical bandwidth utilization. Missing measurements stay blank.

| Operation | Reader, ms | Writer, ms | Compute, ms |
|---|---:|---:|---:|
| MatmulDeviceOperation | 18.9829 | 16.8458 | 18.7345 |
| SdpaDecodeDeviceOperation | 12.6211 | 12.2498 | 12.6128 |
| GenericOpDeviceOperation | 9.1816 | 7.0056 | 8.8659 |
| AllReduceAsyncDeviceOperation | 1.6599 | 2.5905 | 0.1447 |
| BinaryNgDeviceOperation | 1.2222 | 0.7455 | 1.1316 |
| LayerNormDeviceOperation | 0.9531 | 0.5148 | 0.9828 |
| ReshapeViewDeviceOperation | 0.9561 | 0.9011 | - |
| SliceDeviceOperation | 0.9429 | 0.8119 | - |
| ReshardDeviceOperation | 0.1980 | 0.5176 | - |
| UnaryDeviceOperation | 0.3677 | 0.2274 | 0.3179 |
| TopkLargeIndicesDeviceOperation | 0.2723 | 0.2654 | 0.2718 |
| ConcatDeviceOperation | 0.2388 | 0.2289 | - |
| ShardedToInterleavedDeviceOperation | 0.2238 | 0.0841 | - |
| TransposeDeviceOperation | 0.2042 | 0.1531 | - |
| InterleavedToShardedDeviceOperation | 0.1976 | 0.1977 | - |
| RotaryEmbeddingDeviceOperation | 0.1318 | 0.1155 | 0.1226 |
| FillPadDeviceOperation | 0.1083 | 0.0565 | 0.1022 |
| TypecastDeviceOperation | 0.0870 | 0.0640 | 0.0494 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 0.0735 | 0.0729 | - |
| PagedFusedUpdateCacheDeviceOperation | 0.0707 | 0.0457 | 0.0525 |
| TopkRoutePrepDeviceOperation | 0.0243 | 0.0219 | 0.0227 |
| SamplingDeviceOperation | 0.0215 | 0.0159 | 0.0203 |
| TopkRouteFinishDeviceOperation | 0.0205 | 0.0201 | - |
| AllGatherAsyncDeviceOperation | 0.0175 | 0.0175 | - |
| AllGatherDeviceOperation | 0.0151 | 0.0189 | - |
| ManualSeedDeviceOperation | - | 0.0172 | 0.0186 |
| TilizeWithValPaddingDeviceOperation | 0.0152 | 0.0139 | 0.0005 |
| EmbeddingsDeviceOperation | 0.0070 | 0.0060 | - |
| CopyDeviceOperation | 0.0062 | 0.0060 | - |
| UntilizeDeviceOperation | 0.0042 | 0.0017 | 0.0021 |
| ReduceDeviceOperation | 0.0039 | 0.0017 | 0.0034 |
| PlusOneDeviceOperation | - | 0.0031 | - |
| IndexedFillDeviceOperation | 0.0029 | 0.0026 | - |

## Disjoint firmware timelines

The following family durations add within each listed replay only.

| Family | Replay 0, ms | Replay 1, ms | Replay 2, ms |
|---|---:|---:|---:|
| attention | 0.0618 | 0.0625 | 0.0630 |
| cache_update | 0.0220 | 0.0220 | 0.0219 |
| collectives | 0.1888 | 0.1889 | 0.2133 |
| custom_gdn_conv_gated_norm | 8.2130 | 8.2135 | 8.2004 |
| elementwise_and_typecast | 1.4438 | 1.4426 | 1.4390 |
| layout | 3.2153 | 3.2207 | 3.2033 |
| matmul | 0.6957 | 0.6969 | 0.6834 |
| normalization | 0.6866 | 0.6861 | 0.6883 |
| other | 0.0140 | 0.0139 | 0.0147 |
| overlap | 36.8503 | 36.8677 | 36.8884 |
| rotary | 0.0277 | 0.0282 | 0.0271 |
| sampling | 0.3721 | 0.3713 | 0.3702 |
| uncovered_gap | 0.2007 | 0.1977 | 0.2021 |

## Matmul weight-byte estimates

One read of each declared padded BFP8 weight tile (1088 B/32x32); 512 GB/s/chip assumed. Extra transactions, activation traffic and compute excluded.
These are not physical DRAM counters or a calibrated compute roofline.

| Projection | Stored K x N | Calls | Kernel sum, ms | Weight GB/s | Assumed peak fraction |
|---|---:|---:|---:|---:|---:|
| Attention/GDN output projection | 1536 x 5120 | 64 | 1.843 | 290.1 | 56.7% |
| MLP down | 4352 x 5120 | 64 | 4.376 | 346.2 | 67.6% |
| Full-attention packed projection | 5120 x 3584 | 16 | 0.875 | 356.4 | 69.6% |
| GDN packed projection | 5120 x 4608 | 48 | 3.373 | 356.7 | 69.7% |
| MLP gate/up | 5120 x 9216 | 64 | 7.610 | 421.6 | 82.3% |
| Vocabulary head tail chunk | 5120 x 12928 | 1 | 0.195 | 360.7 | 70.4% |
| Vocabulary head full chunk | 5120 x 16384 | 3 | 0.712 | 375.4 | 73.3% |
