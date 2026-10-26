# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 14.88 | 5 | 1 | 14.9 |
| layer_mlstm | 73.31 | 91 | 6 | 439.9 |
| slstm_pre | 24.49 | 4 | 6 | 146.9 |
| slstm_first | 0.55 | 28 | 6 | 3.3 |
| slstm_steps | 12.58 | 621 | 24 | 301.8 |
| slstm_post | 34.36 | 15 | 6 | 206.1 |
| head | 16.61 | 7 | 1 | 16.6 |

Estimated full-model device time: **1129.6 ms** over 15744 ops.

Category share of the reduced window: Compute 78.4%, TM 16.8%, Other 4.7%; matmul weighted FLOPs utilization 19.9%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 40.5 | 71.66 | 47 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 25.7 | 45.42 | 330 |
| UnaryDeviceOperation (in0:dram_interleaved) | 8.3 | 14.76 | 207 |
| PermuteDeviceOperation (in0:dram_interleaved) | 6.3 | 11.20 | 4 |
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 4.7 | 8.38 | 2 |
| SliceDeviceOperation (in0:dram_interleaved) | 4.1 | 7.25 | 130 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 2.7 | 4.80 | 6 |
| ConcatDeviceOperation (in0:dram_interleaved) | 1.9 | 3.45 | 5 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 1.1 | 1.89 | 1 |
| ReduceDeviceOperation (in0:dram_interleaved) | 0.9 | 1.55 | 4 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.8 | 1.37 | 2 |
| UntilizeWithUnpaddingDeviceOperation (in0:dram_interleaved) | 0.7 | 1.31 | 1 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 0.7 | 1.28 | 1 |
| TransposeDeviceOperation (in0:dram_interleaved) | 0.6 | 1.02 | 3 |
| AccumulationDeviceOperation (in0:dram_interleaved) | 0.2 | 0.44 | 1 |
| UntilizeDeviceOperation (in0:dram_interleaved) | 0.2 | 0.44 | 1 |
| FillPadDeviceOperation (in0:dram_interleaved) | 0.2 | 0.33 | 1 |
| TypecastDeviceOperation (in0:dram_interleaved) | 0.1 | 0.23 | 25 |
