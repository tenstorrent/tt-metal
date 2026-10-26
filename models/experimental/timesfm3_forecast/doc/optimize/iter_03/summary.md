# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 3.89 | 4 | 1 | 3.9 |
| layer | 21.84 | 11 | 19 | 414.9 |
| cut | 14.86 | 24 | 1 | 14.9 |
| head | 0.99 | 3 | 1 | 1.0 |

Estimated full-model device time: **434.7 ms** over 240 ops.
Measured: device 423.6 ms (estimate off by +2.6%), steady wall 613.2 ms, pcc_normalized 0.99994.

Category share of the reduced window: Other 79.2%, Compute 13.5%, TM 7.3%; matmul weighted FLOPs utilization 16.9%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 42.8 | 17.78 | 13 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 31.6 | 13.14 | 5 |
| SDPAOperation (in0:dram_interleaved) | 8.2 | 3.43 | 2 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 4.8 | 2.00 | 2 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 4.2 | 1.73 | 4 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 2.9 | 1.22 | 5 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 2.4 | 1.01 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 2.2 | 0.91 | 6 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.7 | 0.30 | 2 |
| MatmulDeviceOperation (in0:dram_interleaved) | 0.1 | 0.04 | 1 |
