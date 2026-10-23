# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 7.85 | 5 | 1 | 7.9 |
| layer_time | 50.98 | 18 | 16 | 815.7 |
| layer_group | 28.16 | 11 | 7 | 197.2 |
| last | 29.43 | 15 | 1 | 29.4 |
| head | 0.55 | 11 | 1 | 0.5 |

Estimated full-model device time: **1050.7 ms** over 396 ops.
Measured: device 1056.4 ms (estimate off by -0.5%), steady wall 1201.0 ms, pcc_normalized 1.00000.

Category share of the reduced window: Compute 95.5%, TM 3.7%, Other 0.8%; matmul weighted FLOPs utilization 27.3%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 60.4 | 70.68 | 22 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 15.2 | 17.73 | 14 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 7.6 | 8.89 | 9 |
| UnaryDeviceOperation (in0:dram_interleaved) | 7.1 | 8.35 | 6 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 3.6 | 4.23 | 2 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 2.3 | 2.73 | 1 |
| SDPAOperation (in0:dram_interleaved) | 1.6 | 1.87 | 1 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 0.8 | 0.92 | 1 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 0.8 | 0.91 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.3 | 0.38 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.2 | 0.24 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.0 | 0.05 | 1 |
