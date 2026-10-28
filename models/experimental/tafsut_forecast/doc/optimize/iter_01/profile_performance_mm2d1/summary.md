# Round summary: paper / performance

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 12.86 | 8 | 1 | 12.9 |
| layer | 25.36 | 11 | 13 | 329.7 |
| layer_last | 16.88 | 13 | 1 | 16.9 |
| head | 5.70 | 6 | 1 | 5.7 |

Estimated full-model device time: **365.2 ms** over 170 ops.

Category share of the reduced window: Compute 84.2%, Other 10.7%, TM 5.1%; matmul weighted FLOPs utilization 29.0%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 54.5 | 33.15 | 14 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 13.2 | 8.01 | 9 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 10.7 | 6.53 | 2 |
| SDPAOperation (in0:dram_interleaved) | 7.1 | 4.34 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 6.0 | 3.64 | 5 |
| UnaryDeviceOperation (in0:dram_interleaved) | 3.3 | 2.03 | 1 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 2.1 | 1.26 | 2 |
| ConcatDeviceOperation (in0:dram_interleaved) | 1.7 | 1.03 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 1.3 | 0.81 | 2 |
