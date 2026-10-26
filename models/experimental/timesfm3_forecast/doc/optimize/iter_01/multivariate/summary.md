# Round summary: multivariate / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 3.92 | 4 | 1 | 3.9 |
| layer | 140.70 | 30 | 20 | 2813.9 |
| head | 1.77 | 5 | 1 | 1.8 |

Estimated full-model device time: **2819.6 ms** over 609 ops.

Category share of the reduced window: TM 64.7%, Compute 28.9%, Other 6.3%; matmul weighted FLOPs utilization 50.2%.

| op type | % | device ms | count |
|---|---|---|---|
| TransposeDeviceOperation (in0:dram_interleaved) | 26.6 | 38.90 | 2 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 18.7 | 27.32 | 3 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 10.5 | 15.43 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 9.7 | 14.15 | 10 |
| FillPadDeviceOperation (in0:dram_interleaved) | 8.6 | 12.60 | 1 |
| MatmulDeviceOperation (in0:dram_interleaved) | 6.9 | 10.15 | 3 |
| SDPAOperation (in0:dram_interleaved) | 6.3 | 9.30 | 2 |
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 5.8 | 8.46 | 7 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 3.0 | 4.46 | 4 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 3.0 | 4.31 | 2 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 0.5 | 0.76 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.3 | 0.48 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.0 | 0.05 | 1 |
