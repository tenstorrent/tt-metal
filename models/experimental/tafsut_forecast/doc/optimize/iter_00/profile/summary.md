# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 19.02 | 8 | 1 | 19.0 |
| layer | 121.92 | 20 | 13 | 1584.9 |
| layer_last | 83.72 | 22 | 1 | 83.7 |
| head | 7.70 | 7 | 1 | 7.7 |

Estimated full-model device time: **1695.3 ms** over 297 ops.

Category share of the reduced window: Compute 96.0%, TM 4.0%; matmul weighted FLOPs utilization 11.3%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 78.5 | 182.42 | 18 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 10.6 | 24.54 | 21 |
| UnaryDeviceOperation (in0:dram_interleaved) | 3.5 | 8.13 | 4 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 2.7 | 6.18 | 2 |
| SDPAOperation (in0:dram_interleaved) | 1.9 | 4.34 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 1.6 | 3.64 | 5 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 0.5 | 1.26 | 2 |
| ConcatDeviceOperation (in0:dram_interleaved) | 0.4 | 1.03 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.3 | 0.81 | 2 |
