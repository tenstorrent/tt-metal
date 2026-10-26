# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 3.81 | 4 | 1 | 3.8 |
| layer | 30.06 | 20 | 20 | 601.2 |
| head | 1.76 | 5 | 1 | 1.8 |

Estimated full-model device time: **606.7 ms** over 409 ops.
Measured: device 606.9 ms (estimate off by -0.0%), steady wall 805.2 ms, pcc_normalized 0.99993.

Category share of the reduced window: Compute 63.4%, Other 25.3%, TM 11.4%; matmul weighted FLOPs utilization 50.0%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 23.1 | 8.24 | 7 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 19.1 | 6.80 | 8 |
| MatmulDeviceOperation (in0:dram_interleaved) | 14.3 | 5.10 | 2 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 12.5 | 4.47 | 4 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 12.1 | 4.31 | 2 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 8.7 | 3.09 | 1 |
| SDPAOperation (in0:dram_interleaved) | 5.3 | 1.90 | 1 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 2.1 | 0.76 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 1.4 | 0.48 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 1.2 | 0.43 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.1 | 0.05 | 1 |
