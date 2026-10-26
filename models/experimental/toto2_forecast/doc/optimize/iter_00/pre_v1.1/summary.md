# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 61.97 | 5 | 1 | 62.0 |
| layer_time | 157.53 | 18 | 46 | 7246.3 |
| last_time | 75.97 | 22 | 1 | 76.0 |
| layer_variate | 1.82 | 13 | 1 | 1.8 |
| head | 1.07 | 6 | 1 | 1.1 |

Estimated full-model device time: **7387.2 ms** over 874 ops.
Measured: device 7433.2 ms (estimate off by -0.6%), steady wall 7765.4 ms, pcc_normalized 0.99990.

Category share of the reduced window: Compute 93.8%, TM 5.6%, Other 0.6%; matmul weighted FLOPs utilization 26.8%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 69.9 | 208.64 | 21 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 8.4 | 25.21 | 17 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 5.8 | 17.30 | 4 |
| UnaryDeviceOperation (in0:dram_interleaved) | 5.2 | 15.51 | 5 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 3.9 | 11.48 | 2 |
| SDPAOperation (in0:dram_interleaved) | 2.4 | 7.17 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 2.0 | 5.94 | 7 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.3 | 3.92 | 2 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 0.6 | 1.88 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.2 | 0.75 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.1 | 0.45 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.0 | 0.09 | 1 |
