# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 7.85 | 5 | 1 | 7.8 |
| layer_time | 51.56 | 18 | 16 | 825.0 |
| layer_group | 28.26 | 11 | 7 | 197.8 |
| last | 29.26 | 15 | 1 | 29.3 |
| head | 0.54 | 11 | 1 | 0.5 |

Estimated full-model device time: **1060.5 ms** over 396 ops.
Measured: device 1056.8 ms (estimate off by +0.3%), steady wall 1547.8 ms, pcc_normalized 0.99997.

Category share of the reduced window: Compute 95.6%, TM 3.7%, Other 0.8%; matmul weighted FLOPs utilization 27.3%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 60.1 | 70.66 | 22 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 15.1 | 17.71 | 14 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 7.6 | 8.90 | 9 |
| UnaryDeviceOperation (in0:dram_interleaved) | 7.5 | 8.87 | 6 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 3.6 | 4.24 | 2 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 2.3 | 2.72 | 1 |
| SDPAOperation (in0:dram_interleaved) | 1.6 | 1.87 | 1 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 0.8 | 0.92 | 1 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 0.8 | 0.91 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.3 | 0.38 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.2 | 0.24 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.0 | 0.05 | 1 |
