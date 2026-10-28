# Round summary: paper / performance

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 18.79 | 8 | 1 | 18.8 |
| layer | 65.36 | 12 | 13 | 849.7 |
| layer_last | 28.13 | 14 | 1 | 28.1 |
| head | 7.66 | 7 | 1 | 7.7 |

Estimated full-model device time: **904.3 ms** over 185 ops.

Category share of the reduced window: Compute 92.0%, Other 5.4%, TM 2.6%; matmul weighted FLOPs utilization 9.7%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 72.0 | 86.36 | 14 |
| UnaryDeviceOperation (in0:dram_interleaved) | 6.8 | 8.14 | 4 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 6.7 | 8.01 | 9 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 5.4 | 6.51 | 2 |
| SDPAOperation (in0:dram_interleaved) | 3.5 | 4.22 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 3.0 | 3.61 | 5 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.0 | 1.25 | 2 |
| ConcatDeviceOperation (in0:dram_interleaved) | 0.8 | 1.03 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.7 | 0.81 | 2 |
