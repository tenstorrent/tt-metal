# Complete operator profile

Full 64-layer model plus split device sampler, real weights and synthetic populated caches; three restored-state trace replays, not natural-prompt accuracy or unprofiled serving throughput

Context 32768, batch 16, 38 operation types.
Unprofiled restored-cache step 67.240 ms; profiled 72.259 ms.
Whole-step profiler overhead 7.46%; this is not HTTP/natural-prompt throughput.

Each rank's firmware span is partitioned by op family with overlap and gaps explicit. Use each replay's longest-rank timeline; do not sum rank times or family medians. Firmware includes waits. Profile overhead is measured for the full step, not per operation.

| Operation | Calls, range | Median kernel sum, ms |
|---|---:|---:|
| MatmulDeviceOperation | 260-260 | 18.8889 |
| SdpaDecodeDeviceOperation | 16-16 | 12.6328 |
| TilizeWithValPaddingDeviceOperation | 387-387 | 5.8631 |
| GenericOpDeviceOperation | 96-96 | 5.2891 |
| SliceDeviceOperation | 1073-1073 | 4.9875 |
| ReshapeViewDeviceOperation | 322-322 | 3.7255 |
| UntilizeWithUnpaddingDeviceOperation | 480-480 | 3.0411 |
| AllReduceAsyncDeviceOperation | 128-128 | 2.6101 |
| BinaryNgDeviceOperation | 364-364 | 1.9039 |
| FillPadDeviceOperation | 161-161 | 1.6678 |
| SigmoidGatedRmsNormOperation | 48-48 | 1.5748 |
| LayerNormDeviceOperation | 161-161 | 1.0847 |
| QkvCausalConv1dSiluOperation | 48-48 | 1.0668 |
| TypecastDeviceOperation | 244-244 | 0.6681 |
| ConcatDeviceOperation | 177-177 | 0.6174 |
| ReshardDeviceOperation | 193-193 | 0.5421 |
| UnaryDeviceOperation | 97-97 | 0.4595 |
| UntilizeCodegenDeviceOperation | 144-144 | 0.4236 |
| TransposeDeviceOperation | 112-112 | 0.3110 |
| TopkLargeIndicesDeviceOperation | 1-1 | 0.2721 |
| ShardedToInterleavedDeviceOperation | 164-164 | 0.2428 |
| InterleavedToShardedDeviceOperation | 146-146 | 0.2355 |
| CopyDeviceOperation | 49-49 | 0.1889 |
| PadDeviceOperation | 48-48 | 0.1683 |
| RotaryEmbeddingDeviceOperation | 32-32 | 0.1369 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 16-16 | 0.0735 |
| PagedFusedUpdateCacheDeviceOperation | 16-16 | 0.0706 |
| AllGatherDeviceOperation | 2-2 | 0.0258 |
| TopkRoutePrepDeviceOperation | 1-1 | 0.0242 |
| SamplingDeviceOperation | 1-1 | 0.0215 |
| TopkRouteFinishDeviceOperation | 1-1 | 0.0204 |
| AllGatherAsyncDeviceOperation | 1-1 | 0.0191 |
| ManualSeedDeviceOperation | 1-1 | 0.0190 |
| EmbeddingsDeviceOperation | 3-3 | 0.0070 |
| UntilizeDeviceOperation | 1-1 | 0.0043 |
| ReduceDeviceOperation | 2-2 | 0.0042 |
| PlusOneDeviceOperation | 4-4 | 0.0031 |
| IndexedFillDeviceOperation | 1-1 | 0.0029 |

## RISC time including waits

These durations are not active compute or physical bandwidth utilization. Missing measurements stay blank.

| Operation | Reader, ms | Writer, ms | Compute, ms |
|---|---:|---:|---:|
| MatmulDeviceOperation | 18.8879 | 16.7341 | 18.6370 |
| SdpaDecodeDeviceOperation | 12.6274 | 12.2575 | 12.6190 |
| TilizeWithValPaddingDeviceOperation | 5.8486 | 4.9651 | 4.0328 |
| GenericOpDeviceOperation | 5.2883 | 3.8475 | 5.1034 |
| SliceDeviceOperation | 4.9819 | 4.6765 | - |
| ReshapeViewDeviceOperation | 3.7211 | 2.9391 | - |
| UntilizeWithUnpaddingDeviceOperation | 3.0330 | 2.2793 | 2.1257 |
| AllReduceAsyncDeviceOperation | 1.6523 | 2.5539 | 0.1437 |
| BinaryNgDeviceOperation | 1.8666 | 1.3428 | 1.7564 |
| FillPadDeviceOperation | 1.6626 | 0.5881 | 1.5970 |
| SigmoidGatedRmsNormOperation | 1.5746 | 0.8219 | 1.5427 |
| LayerNormDeviceOperation | 0.9472 | 0.4820 | 0.9792 |
| QkvCausalConv1dSiluOperation | 1.0668 | 0.5554 | 1.0337 |
| TypecastDeviceOperation | 0.6589 | 0.5086 | 0.2417 |
| ConcatDeviceOperation | 0.6169 | 0.5496 | - |
| ReshardDeviceOperation | 0.2105 | 0.5014 | - |
| UnaryDeviceOperation | 0.4592 | 0.3217 | 0.4116 |
| UntilizeCodegenDeviceOperation | 0.4233 | 0.1836 | 0.1156 |
| TransposeDeviceOperation | 0.3059 | 0.2308 | 0.1070 |
| TopkLargeIndicesDeviceOperation | 0.2721 | 0.2653 | 0.2717 |
| ShardedToInterleavedDeviceOperation | 0.2426 | 0.0834 | - |
| InterleavedToShardedDeviceOperation | 0.2313 | 0.2317 | - |
| CopyDeviceOperation | 0.1888 | 0.1655 | - |
| PadDeviceOperation | 0.1683 | 0.1521 | - |
| RotaryEmbeddingDeviceOperation | 0.1315 | 0.1153 | 0.1224 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 0.0735 | 0.0730 | - |
| PagedFusedUpdateCacheDeviceOperation | 0.0706 | 0.0457 | 0.0525 |
| AllGatherDeviceOperation | 0.0224 | 0.0258 | - |
| TopkRoutePrepDeviceOperation | 0.0242 | 0.0218 | 0.0226 |
| SamplingDeviceOperation | 0.0214 | 0.0160 | 0.0203 |
| TopkRouteFinishDeviceOperation | 0.0204 | 0.0200 | - |
| AllGatherAsyncDeviceOperation | 0.0174 | 0.0175 | - |
| ManualSeedDeviceOperation | - | 0.0176 | 0.0190 |
| EmbeddingsDeviceOperation | 0.0070 | 0.0060 | - |
| UntilizeDeviceOperation | 0.0042 | 0.0017 | 0.0021 |
| ReduceDeviceOperation | 0.0039 | 0.0017 | 0.0034 |
| PlusOneDeviceOperation | - | 0.0031 | - |
| IndexedFillDeviceOperation | 0.0029 | 0.0027 | - |

## Disjoint firmware timelines

The following family durations add within each listed replay only.

| Family | Replay 0, ms | Replay 1, ms | Replay 2, ms |
|---|---:|---:|---:|
| attention | 0.0905 | 0.0882 | 0.0893 |
| cache_update | 0.0219 | 0.0218 | 0.0219 |
| collectives | 0.2077 | 0.2037 | 0.2054 |
| custom_gdn_conv_gated_norm | 4.7370 | 4.7351 | 4.7349 |
| elementwise_and_typecast | 2.2866 | 2.2938 | 2.2919 |
| layout | 21.7239 | 21.7119 | 21.7345 |
| matmul | 0.8219 | 0.8094 | 0.8241 |
| normalization | 0.6817 | 0.6838 | 0.6840 |
| other | 0.0139 | 0.0146 | 0.0137 |
| overlap | 40.6233 | 40.6574 | 40.6085 |
| rotary | 0.0270 | 0.0265 | 0.0279 |
| sampling | 0.3711 | 0.3712 | 0.3706 |
| uncovered_gap | 0.3419 | 0.3395 | 0.3386 |

## Matmul weight-byte estimates

One read of each declared padded BFP8 weight tile (1088 B/32x32); 512 GB/s/chip assumed. Extra transactions, activation traffic and compute excluded.
These are not physical DRAM counters or a calibrated compute roofline.

| Projection | Stored K x N | Calls | Kernel sum, ms | Weight GB/s | Assumed peak fraction |
|---|---:|---:|---:|---:|---:|
| Attention/GDN output projection | 1536 x 5120 | 64 | 1.833 | 291.8 | 57.0% |
| MLP down | 4352 x 5120 | 64 | 4.347 | 348.6 | 68.1% |
| Full-attention packed projection | 5120 x 3584 | 16 | 0.870 | 358.7 | 70.1% |
| GDN packed projection | 5120 x 4608 | 48 | 3.352 | 358.9 | 70.1% |
| MLP gate/up | 5120 x 9216 | 64 | 7.593 | 422.6 | 82.5% |
| Vocabulary head tail chunk | 5120 x 12928 | 1 | 0.194 | 362.8 | 70.9% |
| Vocabulary head full chunk | 5120 x 16384 | 3 | 0.703 | 380.4 | 74.3% |
