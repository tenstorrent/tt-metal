# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 7.81 | 5 | 1 | 7.8 |
| layer_time | 28.35 | 12 | 15 | 425.2 |
| layer_group | 12.41 | 5 | 7 | 86.9 |
| cut | 20.52 | 16 | 1 | 20.5 |
| last | 0.43 | 5 | 1 | 0.4 |
| head | 0.55 | 11 | 1 | 0.6 |

Estimated full-model device time: **541.4 ms** over 252 ops.
Measured: device 541.8 ms (estimate off by -0.1%), steady wall 685.1 ms, pcc_normalized 0.99997.

Category share of the reduced window: Other 46.2%, Compute 45.0%, TM 8.8%; matmul weighted FLOPs utilization 29.1%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 44.9 | 31.49 | 14 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 15.3 | 10.75 | 13 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 12.2 | 8.56 | 4 |
| MatmulDeviceOperation (in0:dram_interleaved) | 8.1 | 5.69 | 7 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 7.8 | 5.47 | 2 |
| SDPAOperation (in0:dram_interleaved) | 5.6 | 3.90 | 2 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 2.2 | 1.54 | 5 |
| UnaryDeviceOperation (in0:dram_interleaved) | 1.6 | 1.11 | 3 |
| UntilizeCodegenDeviceOperation (in0:dram_interleaved) | 1.3 | 0.90 | 1 |
| TilizeWithValPaddingDeviceOperation (in0:dram_interleaved) | 0.5 | 0.37 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.4 | 0.25 | 1 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.1 | 0.05 | 1 |
