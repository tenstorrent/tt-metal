# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 66.36 | 5 | 1 | 66.4 |
| layer_mlstm | 274.09 | 142 | 6 | 1644.5 |
| slstm_pre | 54.49 | 4 | 6 | 326.9 |
| slstm_first | 0.14 | 9 | 6 | 0.9 |
| slstm_steps | 38.80 | 1917 | 24 | 931.2 |
| slstm_post | 129.66 | 20 | 6 | 777.9 |
| head | 87.13 | 7 | 1 | 87.1 |

Estimated full-model device time: **3835.0 ms** over 47070 ops.
Measured: device 3856.9 ms (estimate off by -0.6%), steady wall 4518.4 ms, pcc_normalized 0.99991.

Category share of the reduced window: Compute 87.7%, TM 12.3%; matmul weighted FLOPs utilization 9.7%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 57.0 | 371.12 | 98 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 20.7 | 134.68 | 927 |
| UnaryDeviceOperation (in0:dram_interleaved) | 7.4 | 47.88 | 593 |
| PermuteDeviceOperation (in0:dram_interleaved) | 4.3 | 28.18 | 4 |
| SliceDeviceOperation (in0:dram_interleaved) | 3.1 | 20.12 | 382 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 1.9 | 12.07 | 6 |
| ConcatDeviceOperation (in0:dram_interleaved) | 1.8 | 11.91 | 7 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 0.7 | 4.70 | 1 |
| ReduceDeviceOperation (in0:dram_interleaved) | 0.6 | 3.89 | 4 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.5 | 3.45 | 2 |
| UntilizeWithUnpaddingDeviceOperation (in0:dram_interleaved) | 0.5 | 3.27 | 1 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 0.5 | 3.20 | 1 |
| TransposeDeviceOperation (in0:dram_interleaved) | 0.4 | 2.53 | 3 |
| AccumulationDeviceOperation (in0:dram_interleaved) | 0.2 | 1.10 | 1 |
| UntilizeDeviceOperation (in0:dram_interleaved) | 0.2 | 1.09 | 1 |
| FillPadDeviceOperation (in0:dram_interleaved) | 0.1 | 0.81 | 1 |
| TypecastDeviceOperation (in0:dram_interleaved) | 0.1 | 0.66 | 72 |
