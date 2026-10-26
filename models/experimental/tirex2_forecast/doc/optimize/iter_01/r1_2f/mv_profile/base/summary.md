# Round summary: multivariate / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 92.73 | 5 | 1 | 92.7 |
| layer_mlstm | 454.00 | 159 | 6 | 2724.0 |
| slstm_pre | 102.79 | 3 | 6 | 616.8 |
| slstm_first | 0.14 | 9 | 6 | 0.8 |
| slstm_steps | 38.79 | 1917 | 24 | 930.9 |
| slstm_post | 306.74 | 37 | 6 | 1840.5 |
| head | 53.45 | 7 | 1 | 53.4 |

Estimated full-model device time: **6259.2 ms** over 47268 ops.

Category share of the reduced window: Compute 84.7%, TM 13.3%, Other 2.0%; matmul weighted FLOPs utilization 9.3%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 65.3 | 684.54 | 106 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 12.2 | 127.37 | 936 |
| PermuteDeviceOperation (in0:dram_interleaved) | 7.3 | 76.18 | 8 |
| UnaryDeviceOperation (in0:dram_interleaved) | 4.8 | 49.77 | 593 |
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 2.0 | 20.82 | 2 |
| SliceDeviceOperation (in0:dram_interleaved) | 1.9 | 20.14 | 382 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 1.8 | 18.83 | 10 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 1.3 | 13.77 | 3 |
| ConcatDeviceOperation (in0:dram_interleaved) | 1.1 | 11.90 | 7 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 0.6 | 6.24 | 3 |
| ReduceDeviceOperation (in0:dram_interleaved) | 0.4 | 3.90 | 4 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.3 | 3.43 | 2 |
| UntilizeWithUnpaddingDeviceOperation (in0:dram_interleaved) | 0.3 | 3.27 | 1 |
| TransposeDeviceOperation (in0:dram_interleaved) | 0.2 | 2.53 | 3 |
| SoftmaxDeviceOperation (in0:dram_interleaved) | 0.2 | 2.33 | 2 |
| UntilizeDeviceOperation (in0:dram_interleaved) | 0.1 | 1.09 | 1 |
| AccumulationDeviceOperation (in0:dram_interleaved) | 0.1 | 1.07 | 1 |
| FillPadDeviceOperation (in0:dram_interleaved) | 0.1 | 0.83 | 1 |
| TypecastDeviceOperation (in0:dram_interleaved) | 0.1 | 0.66 | 72 |
