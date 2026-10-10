# B16/32K measured operator inventory

B16/32K TP4, BFP8 weights/KV, FP32 recurrent state; 64 layers, 4 ranks, 3 replays

All 38 operation types and 5001 device-operation records per rank/replay are included.
These are not program counts. Profiler overhead is about 7.46%; do not subtract it uniformly.
Family medians are not a disjoint wall-time budget; RISC durations include waiting.

| Operation | Calls/step | Median kernel sum, ms | Rank/replay range, ms |
|---|---:|---:|---:|
| MatmulDeviceOperation | 260 | 18.8889 | 18.7791-18.9056 |
| SdpaDecodeDeviceOperation | 16 | 12.6328 | 12.1047-12.6357 |
| TilizeWithValPaddingDeviceOperation | 387 | 5.8631 | 5.8588-5.8690 |
| GenericOpDeviceOperation | 96 | 5.2891 | 5.2859-5.2951 |
| SliceDeviceOperation | 1073 | 4.9875 | 4.9635-5.0145 |
| ReshapeViewDeviceOperation | 322 | 3.7255 | 3.7050-3.7424 |
| UntilizeWithUnpaddingDeviceOperation | 480 | 3.0411 | 3.0323-3.0568 |
| AllReduceAsyncDeviceOperation | 128 | 2.6101 | 2.5833-3.3083 |
| BinaryNgDeviceOperation | 364 | 1.9039 | 1.8967-1.9181 |
| FillPadDeviceOperation | 161 | 1.6678 | 1.6649-1.6712 |
| SigmoidGatedRmsNormOperation | 48 | 1.5748 | 1.5596-1.5796 |
| LayerNormDeviceOperation | 161 | 1.0847 | 1.0822-1.0882 |
| QkvCausalConv1dSiluOperation | 48 | 1.0668 | 1.0531-1.0745 |
| TypecastDeviceOperation | 244 | 0.6681 | 0.6644-0.6783 |
| ConcatDeviceOperation | 177 | 0.6174 | 0.6078-0.6288 |
| ReshardDeviceOperation | 193 | 0.5421 | 0.5321-0.5550 |
| UnaryDeviceOperation | 97 | 0.4595 | 0.4504-0.4704 |
| UntilizeCodegenDeviceOperation | 144 | 0.4236 | 0.4228-0.4266 |
| TransposeDeviceOperation | 112 | 0.3110 | 0.3004-0.3159 |
| TopkLargeIndicesDeviceOperation | 1 | 0.2721 | 0.2710-0.2742 |
| ShardedToInterleavedDeviceOperation | 164 | 0.2428 | 0.2406-0.2439 |
| InterleavedToShardedDeviceOperation | 146 | 0.2355 | 0.2307-0.2416 |
| CopyDeviceOperation | 49 | 0.1889 | 0.1807-0.1994 |
| PadDeviceOperation | 48 | 0.1683 | 0.1559-0.1851 |
| RotaryEmbeddingDeviceOperation | 32 | 0.1369 | 0.1355-0.1379 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 16 | 0.0735 | 0.0729-0.0737 |
| PagedFusedUpdateCacheDeviceOperation | 16 | 0.0706 | 0.0702-0.0707 |
| AllGatherDeviceOperation | 2 | 0.0258 | 0.0142-0.0339 |
| TopkRoutePrepDeviceOperation | 1 | 0.0242 | 0.0238-0.0252 |
| SamplingDeviceOperation | 1 | 0.0215 | 0.0214-0.0215 |
| TopkRouteFinishDeviceOperation | 1 | 0.0204 | 0.0202-0.0214 |
| AllGatherAsyncDeviceOperation | 1 | 0.0191 | 0.0161-0.0268 |
| ManualSeedDeviceOperation | 1 | 0.0190 | 0.0189-0.0192 |
| EmbeddingsDeviceOperation | 3 | 0.0070 | 0.0067-0.0078 |
| UntilizeDeviceOperation | 1 | 0.0043 | 0.0043-0.0044 |
| ReduceDeviceOperation | 2 | 0.0042 | 0.0041-0.0043 |
| PlusOneDeviceOperation | 4 | 0.0031 | 0.0031-0.0032 |
| IndexedFillDeviceOperation | 1 | 0.0029 | 0.0028-0.0049 |

## Matmul weight-streaming estimates

One read of every declared padded BFP8 weight tile (1088 B/32x32); assumed 512 GB/s/chip; excludes extra transactions, activations, and compute constraints
This is not measured physical DRAM utilization or a calibrated compute roofline.
Different call shapes have different overheads; padding is included in encoded bytes.

| Projection | Stored K x N | Calls | Kernel sum, ms | Weight GB/s | Fraction of assumed peak |
|---|---:|---:|---:|---:|---:|
| GDN packed projection | 5120 x 4608 | 48 | 3.352 | 358.9 | 70.1% |
| Attention/GDN output projection | 1536 x 5120 | 64 | 1.833 | 291.8 | 57.0% |
| MLP gate/up | 5120 x 9216 | 64 | 7.593 | 422.6 | 82.5% |
| MLP down | 4352 x 5120 | 64 | 4.347 | 348.6 | 68.1% |
| Full-attention packed projection | 5120 x 3584 | 16 | 0.870 | 358.7 | 70.1% |
| Vocabulary head full chunk | 5120 x 16384 | 3 | 0.703 | 380.4 | 74.3% |
| Vocabulary head tail chunk | 5120 x 12928 | 1 | 0.194 | 362.8 | 70.9% |

Aggregate: 7.111516 GB / 18.889 ms = 376.5 GB/s (73.5%).
The encoded-weight-only floor is 13.890 ms at peak, or 15.433 ms at 90%.
Those lower bounds omit necessary work and are not promised kernel timings.
