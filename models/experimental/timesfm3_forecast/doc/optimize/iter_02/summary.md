# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 3.80 | 4 | 1 | 3.8 |
| layer | 25.91 | 15 | 20 | 518.2 |
| head | 1.77 | 5 | 1 | 1.8 |

Estimated full-model device time: **523.8 ms** over 309 ops.
Measured: device 501.6 ms (estimate off by +4.4%), steady wall 686.7 ms, pcc_normalized 0.99994.

Category share of the reduced window: Other 59.1%, Compute 28.1%, TM 12.9%; matmul weighted FLOPs utilization 17.1%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 41.3 | 13.00 | 8 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 15.3 | 4.83 | 3 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 9.8 | 3.09 | 1 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 9.7 | 3.07 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 8.6 | 2.71 | 3 |
| SDPAOperation (in0:dram_interleaved) | 6.0 | 1.90 | 1 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 3.5 | 1.11 | 1 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 2.4 | 0.76 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 1.5 | 0.48 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 1.4 | 0.43 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.2 | 0.05 | 1 |
| MatmulDeviceOperation (in0:dram_interleaved) | 0.1 | 0.04 | 1 |
