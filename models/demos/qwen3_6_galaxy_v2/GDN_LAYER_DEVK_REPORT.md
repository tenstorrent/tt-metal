# Qwen3.6 GDN (linear-attention) decode layer — device-kernel report

Per-layer device-kernel (Tracy eager, in-flow, max-over-32-dev, wrap-filtered).
NOTE: eager profiling inflates absolutes ~2x vs the traced production step (78ms est vs 37ms real);
RELATIVE ranking is accurate. CCL ops marked [CCL].

| op | occ/layer | devk us/layer | CCL? |
|---|---:|---:|:--:|
| RMSAllGatherDeviceOperation | 3.0 | 334.4 | CCL |
| MatmulDeviceOperation | 9.0 | 142.7 |  |
| ReduceScatterMinimalAsyncDeviceOperation | 4.0 | 139.1 | CCL |
| AllGatherAsyncDeviceOperation | 4.0 | 102.9 | CCL |
| AllReduceAsyncDeviceOperation | 1.0 | 68.6 | CCL |
| ReduceScatterDeviceOperation | 1.0 | 60.3 | CCL |
| AllGatherDeviceOperation | 1.0 | 47.1 | CCL |
| AllBroadcastDeviceOperation | 1.0 | 36.7 | CCL |
| BinaryNgDeviceOperation | 16.0 | 33.9 |  |
| TilizeWithValPaddingDeviceOperation | 7.0 | 25.7 |  |
| ReshapeViewDeviceOperation | 9.0 | 18.1 |  |
| TilizeDeviceOperation | 1.0 | 12.8 |  |
| LayerNormDeviceOperation | 3.0 | 12.4 |  |
| SliceDeviceOperation | 15.0 | 12.4 |  |
| TypecastDeviceOperation | 11.0 | 12.0 |  |
| UntilizeWithUnpaddingDeviceOperation | 5.0 | 10.3 |  |
| UnaryDeviceOperation | 6.0 | 8.2 |  |
| GenericOpDeviceOperation | 1.0 | 6.4 |  |
| TernaryDeviceOperation | 1.0 | 5.8 |  |
| CopyDeviceOperation | 3.0 | 5.0 |  |
| ConcatDeviceOperation | 4.0 | 4.1 |  |
| ShardedToInterleavedDeviceOperation | 4.0 | 3.0 |  |
| TransposeDeviceOperation | 2.0 | 3.0 |  |
| MorehSumOperation | 1.0 | 2.5 |  |
| RepeatDeviceOperation | 1.0 | 2.1 |  |
| **TOTAL** |  | **1109.6** |  |

**CCL = 789 us/layer (71% of layer); compute/other = 320 us.**
