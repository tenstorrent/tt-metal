# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 66.37 | 5 | 1 | 66.4 |
| layer_mlstm | 275.45 | 142 | 6 | 1652.7 |
| slstm_pre | 54.57 | 4 | 6 | 327.4 |
| slstm_first | 0.14 | 9 | 6 | 0.9 |
| slstm_steps | 38.80 | 1917 | 24 | 931.2 |
| slstm_post | 130.84 | 20 | 6 | 785.1 |
| head | 87.75 | 7 | 1 | 87.8 |

Estimated full-model device time: **3851.2 ms** over 47070 ops.
Measured: device 3871.2 ms (estimate off by -0.5%), steady wall 4278.6 ms, pcc_normalized 0.99997.

Category share of the reduced window: Compute 87.8%, TM 12.2%; matmul weighted FLOPs utilization 15.4%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 57.2 | 374.21 | 98 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 20.6 | 134.60 | 927 |
| UnaryDeviceOperation (in0:dram_interleaved) | 7.3 | 48.02 | 593 |
| PermuteDeviceOperation (in0:dram_interleaved) | 4.3 | 28.22 | 4 |
| SliceDeviceOperation (in0:dram_interleaved) | 3.1 | 20.10 | 382 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 1.9 | 12.13 | 6 |
| ConcatDeviceOperation (in0:dram_interleaved) | 1.8 | 11.92 | 7 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 0.7 | 4.71 | 1 |
| ReduceDeviceOperation (in0:dram_interleaved) | 0.6 | 3.88 | 4 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.5 | 3.44 | 2 |
| UntilizeWithUnpaddingDeviceOperation (in0:dram_interleaved) | 0.5 | 3.27 | 1 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 0.5 | 3.22 | 1 |
| TransposeDeviceOperation (in0:dram_interleaved) | 0.4 | 2.52 | 3 |
| AccumulationDeviceOperation (in0:dram_interleaved) | 0.2 | 1.11 | 1 |
| UntilizeDeviceOperation (in0:dram_interleaved) | 0.2 | 1.09 | 1 |
| FillPadDeviceOperation (in0:dram_interleaved) | 0.1 | 0.82 | 1 |
| TypecastDeviceOperation (in0:dram_interleaved) | 0.1 | 0.66 | 72 |
