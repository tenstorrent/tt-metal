# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 12.89 | 8 | 1 | 12.9 |
| layer | 28.53 | 11 | 13 | 371.0 |
| layer_last | 19.27 | 13 | 1 | 19.3 |
| head | 5.65 | 6 | 1 | 5.7 |

Estimated full-model device time: **408.8 ms** over 170 ops.

Category share of the reduced window: Compute 85.4%, Other 9.9%, TM 4.6%; matmul weighted FLOPs utilization 45.7%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 58.2 | 38.61 | 14 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 12.1 | 8.05 | 9 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 9.9 | 6.59 | 2 |
| SDPAOperation (in0:dram_interleaved) | 6.5 | 4.34 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 5.5 | 3.66 | 5 |
| UnaryDeviceOperation (in0:dram_interleaved) | 3.1 | 2.03 | 1 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.9 | 1.25 | 2 |
| ConcatDeviceOperation (in0:dram_interleaved) | 1.5 | 1.02 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 1.2 | 0.81 | 2 |
