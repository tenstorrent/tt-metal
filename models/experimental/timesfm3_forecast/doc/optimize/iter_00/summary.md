# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 9.37 | 5 | 1 | 9.4 |
| layer | 67.85 | 22 | 20 | 1356.9 |
| head | 2.15 | 5 | 1 | 2.1 |

Estimated full-model device time: **1368.4 ms** over 450 ops.
Measured: device 1372.0 ms (estimate off by -0.3%), steady wall 1971.5 ms, pcc_normalized 0.99993.

Category share of the reduced window: Compute 89.5%, TM 9.0%, Other 1.4%; matmul weighted FLOPs utilization 23.4%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 51.9 | 41.21 | 9 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 13.0 | 10.28 | 8 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 9.4 | 7.49 | 2 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 8.3 | 6.63 | 4 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 6.1 | 4.81 | 1 |
| SDPAOperation (in0:dram_interleaved) | 3.5 | 2.81 | 1 |
| UnaryDeviceOperation (in0:dram_interleaved) | 3.3 | 2.61 | 2 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.8 | 1.41 | 1 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 1.4 | 1.14 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.6 | 0.48 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.5 | 0.43 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.1 | 0.05 | 1 |
