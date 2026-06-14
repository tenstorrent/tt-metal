# Qwen3.6 FA (full-attention) decode layer — device-kernel report

Per-layer device-kernel (Tracy eager, in-flow, max-over-32-dev, wrap-filtered).
NOTE: eager profiling inflates absolutes ~2x vs the traced production step (78ms est vs 37ms real);
RELATIVE ranking is accurate. CCL ops marked [CCL].

| op | occ/layer | devk us/layer | CCL? |
|---|---:|---:|:--:|
| RMSAllGatherDeviceOperation | 2.0 | 722.2 | CCL |
| ReduceScatterDeviceOperation | 2.0 | 158.7 | CCL |
| ReduceScatterMinimalAsyncDeviceOperation | 3.0 | 137.7 | CCL |
| MatmulDeviceOperation | 5.0 | 132.1 |  |
| AllGatherDeviceOperation | 2.0 | 112.6 | CCL |
| AllGatherAsyncDeviceOperation | 2.0 | 108.6 | CCL |
| SliceDeviceOperation | 13.0 | 13.0 |  |
| TilizeDeviceOperation | 1.0 | 12.7 |  |
| LayerNormDeviceOperation | 2.0 | 12.6 |  |
| BinaryNgDeviceOperation | 6.0 | 10.4 |  |
| PagedUpdateCacheDeviceOperation | 2.0 | 7.9 |  |
| SdpaDecodeDeviceOperation | 1.0 | 7.8 |  |
| UnaryDeviceOperation | 5.0 | 5.7 |  |
| InterleavedToShardedDeviceOperation | 6.0 | 5.6 |  |
| TransposeDeviceOperation | 3.0 | 4.8 |  |
| ReshapeViewDeviceOperation | 2.0 | 4.5 |  |
| ShardedToInterleavedDeviceOperation | 5.0 | 4.5 |  |
| ConcatDeviceOperation | 4.0 | 4.2 |  |
| TernaryDeviceOperation | 2.0 | 3.6 |  |
| RepeatDeviceOperation | 1.0 | 2.1 |  |
| CloneOperation | 2.0 | 2.0 |  |
| **TOTAL** |  | **1473.3** |  |

**CCL = 1240 us/layer (84% of layer); compute/other = 234 us.**
