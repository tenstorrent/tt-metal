# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 3.87 | 4 | 1 | 3.9 |
| layer | 25.87 | 15 | 19 | 491.5 |
| cut | 18.79 | 28 | 1 | 18.8 |
| head | 0.99 | 3 | 1 | 1.0 |

Estimated full-model device time: **515.1 ms** over 320 ops.

Category share of the reduced window: Other 50.0%, Compute 31.3%, TM 18.6%; matmul weighted FLOPs utilization 16.9%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 36.1 | 17.89 | 13 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 12.5 | 6.18 | 2 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 12.3 | 6.11 | 4 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 9.9 | 4.88 | 3 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 9.5 | 4.71 | 10 |
| SDPAOperation (in0:dram_interleaved) | 6.9 | 3.43 | 2 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 4.0 | 2.00 | 2 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 3.5 | 1.74 | 4 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 2.5 | 1.21 | 5 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 2.0 | 1.01 | 2 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.6 | 0.30 | 2 |
| MatmulDeviceOperation (in0:dram_interleaved) | 0.1 | 0.04 | 1 |
