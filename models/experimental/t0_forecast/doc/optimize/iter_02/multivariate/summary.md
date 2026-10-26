# Round summary: multivariate / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 7.81 | 5 | 1 | 7.8 |
| layer_time | 28.47 | 12 | 15 | 427.1 |
| layer_group | 124.15 | 15 | 7 | 869.0 |
| cut | 20.52 | 16 | 1 | 20.5 |
| last | 4.66 | 17 | 1 | 4.7 |
| head | 0.55 | 11 | 1 | 0.5 |

Estimated full-model device time: **1329.7 ms** over 334 ops.

Category share of the reduced window: TM 53.2%, Compute 27.0%, Other 19.8%; matmul weighted FLOPs utilization 29.1%.

| op type | % | device ms | count |
|---|---|---|---|
| TransposeDeviceOperation (in0:dram_interleaved) | 21.4 | 39.80 | 4 |
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 19.4 | 36.02 | 16 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 15.8 | 29.33 | 7 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 11.9 | 22.07 | 17 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 8.9 | 16.59 | 4 |
| FillPadDeviceOperation (in0:dram_interleaved) | 6.9 | 12.89 | 2 |
| SDPAOperation (in0:dram_interleaved) | 6.0 | 11.23 | 4 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 4.6 | 8.56 | 4 |
| MatmulDeviceOperation (in0:dram_interleaved) | 3.0 | 5.68 | 7 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 0.8 | 1.53 | 5 |
| UnaryDeviceOperation (in0:dram_interleaved) | 0.6 | 1.11 | 3 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 0.5 | 0.91 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.2 | 0.37 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.0 | 0.05 | 1 |
