# Round summary: multivariate / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 3.84 | 4 | 1 | 3.8 |
| layer | 44.64 | 19 | 19 | 848.2 |
| cut | 16.77 | 34 | 1 | 16.8 |
| head | 0.99 | 3 | 1 | 1.0 |

Estimated full-model device time: **869.8 ms** over 402 ops.

Category share of the reduced window: Other 57.5%, TM 28.1%, Compute 14.4%; matmul weighted FLOPs utilization 16.9%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 34.4 | 22.80 | 15 |
| GenericOpDeviceOperation (in0:dram_interleaved) | 20.1 | 13.29 | 5 |
| TransposeDeviceOperation (in0:dram_interleaved) | 17.3 | 11.46 | 4 |
| SDPAOperation (in0:dram_interleaved) | 8.2 | 5.42 | 4 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 4.7 | 3.14 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 4.3 | 2.85 | 10 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 4.0 | 2.68 | 6 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 3.0 | 2.01 | 2 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 1.9 | 1.23 | 7 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 1.5 | 1.01 | 2 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.5 | 0.30 | 2 |
| MatmulDeviceOperation (in0:dram_interleaved) | 0.1 | 0.04 | 1 |
