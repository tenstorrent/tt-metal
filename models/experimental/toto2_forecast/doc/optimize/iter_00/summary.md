# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 49.56 | 5 | 1 | 49.6 |
| layer_time | 169.66 | 18 | 46 | 7804.2 |
| last_time | 84.41 | 22 | 1 | 84.4 |
| layer_variate | 2.12 | 13 | 1 | 2.1 |
| head | 1.25 | 6 | 1 | 1.2 |

Estimated full-model device time: **7941.5 ms** over 874 ops.

Category share of the reduced window: Compute 93.9%, TM 5.4%, Other 0.6%; matmul weighted FLOPs utilization 31.3%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 70.1 | 215.15 | 21 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 8.2 | 25.23 | 17 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 5.6 | 17.30 | 4 |
| UnaryDeviceOperation (in0:dram_interleaved) | 5.2 | 15.86 | 5 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 3.7 | 11.48 | 2 |
| SDPAOperation (in0:dram_interleaved) | 2.9 | 8.93 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 1.9 | 5.95 | 7 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.3 | 3.92 | 2 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 0.6 | 1.88 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.2 | 0.75 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.1 | 0.46 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.0 | 0.09 | 1 |
