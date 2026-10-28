# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 19.03 | 8 | 1 | 19.0 |
| layer | 69.09 | 12 | 13 | 898.1 |
| layer_last | 29.74 | 14 | 1 | 29.7 |
| head | 7.71 | 7 | 1 | 7.7 |

Estimated full-model device time: **954.6 ms** over 185 ops.

Category share of the reduced window: Compute 92.3%, Other 5.3%, TM 2.4%; matmul weighted FLOPs utilization 17.4%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 73.0 | 91.72 | 14 |
| UnaryDeviceOperation (in0:dram_interleaved) | 6.5 | 8.12 | 4 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 6.4 | 8.05 | 9 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 5.3 | 6.62 | 2 |
| SDPAOperation (in0:dram_interleaved) | 3.5 | 4.34 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 2.9 | 3.65 | 5 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.0 | 1.25 | 2 |
| ConcatDeviceOperation (in0:dram_interleaved) | 0.8 | 1.02 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.6 | 0.80 | 2 |
