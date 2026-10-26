# Round summary: multivariate / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 3.87 | 4 | 1 | 3.9 |
| layer | 48.45 | 23 | 20 | 969.0 |
| head | 1.76 | 5 | 1 | 1.8 |

Estimated full-model device time: **974.6 ms** over 469 ops.

Category share of the reduced window: Other 43.1%, TM 33.3%, Compute 23.6%; matmul weighted FLOPs utilization 17.1%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 32.8 | 17.71 | 9 |
| TransposeDeviceOperation (in0:dram_interleaved) | 20.1 | 10.89 | 2 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 11.4 | 6.17 | 2 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 9.0 | 4.84 | 3 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 8.5 | 4.60 | 5 |
| SDPAOperation (in0:dram_interleaved) | 7.2 | 3.90 | 2 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 5.7 | 3.06 | 2 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 2.1 | 1.14 | 2 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 1.4 | 0.76 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.9 | 0.48 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.8 | 0.42 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.1 | 0.05 | 1 |
| MatmulDeviceOperation (in0:dram_interleaved) | 0.1 | 0.04 | 1 |
