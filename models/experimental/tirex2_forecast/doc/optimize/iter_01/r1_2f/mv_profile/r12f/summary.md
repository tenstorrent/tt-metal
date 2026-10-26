# Round summary: multivariate / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 26.41 | 5 | 1 | 26.4 |
| layer_mlstm | 281.20 | 158 | 6 | 1687.2 |
| slstm_pre | 30.35 | 3 | 6 | 182.1 |
| slstm_first | 0.14 | 9 | 6 | 0.8 |
| slstm_steps | 38.79 | 1917 | 24 | 930.9 |
| slstm_post | 192.80 | 36 | 6 | 1156.8 |
| head | 43.73 | 6 | 1 | 43.7 |

Estimated full-model device time: **4028.1 ms** over 47255 ops.

Category share of the reduced window: Compute 56.5%, TM 22.8%, Other 20.7%; matmul weighted FLOPs utilization 9.9%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 26.6 | 162.98 | 90 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 20.7 | 127.15 | 936 |
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 20.7 | 126.72 | 18 |
| PermuteDeviceOperation (in0:dram_interleaved) | 12.4 | 76.22 | 8 |
| UnaryDeviceOperation (in0:dram_interleaved) | 4.9 | 30.27 | 590 |
| SliceDeviceOperation (in0:dram_interleaved) | 3.3 | 20.13 | 382 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 3.1 | 18.90 | 10 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 2.2 | 13.76 | 3 |
| ConcatDeviceOperation (in0:dram_interleaved) | 1.9 | 11.91 | 7 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.0 | 6.25 | 3 |
| ReduceDeviceOperation (in0:dram_interleaved) | 0.6 | 3.90 | 4 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.6 | 3.45 | 2 |
| UntilizeWithUnpaddingDeviceOperation (in0:dram_interleaved) | 0.5 | 3.27 | 1 |
| TransposeDeviceOperation (in0:dram_interleaved) | 0.4 | 2.52 | 3 |
| SoftmaxDeviceOperation (in0:dram_interleaved) | 0.4 | 2.35 | 2 |
| UntilizeDeviceOperation (in0:dram_interleaved) | 0.2 | 1.08 | 1 |
| AccumulationDeviceOperation (in0:dram_interleaved) | 0.2 | 1.06 | 1 |
| FillPadDeviceOperation (in0:dram_interleaved) | 0.1 | 0.83 | 1 |
| TypecastDeviceOperation (in0:dram_interleaved) | 0.1 | 0.66 | 72 |
