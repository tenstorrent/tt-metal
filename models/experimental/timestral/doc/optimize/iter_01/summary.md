# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 9.30 | 6 | 1 | 9.3 |
| layer_time | 27.21 | 17 | 18 | 489.8 |
| layer_time_variate | 33.39 | 23 | 6 | 200.3 |
| head | 0.74 | 7 | 1 | 0.7 |

Estimated full-model device time: **700.2 ms** over 457 ops.
Measured: device 702.0 ms (estimate off by -0.3%), steady wall 836.4 ms, PCC 0.99993 normalized.

Category share of the reduced window: Compute 75.7%, Other 15.5%, TM 8.8%; matmul weighted FLOPs utilization 35.9%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 32.1 | 22.65 | 13 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 23.3 | 16.46 | 14 |
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 15.5 | 10.96 | 8 |
| RotaryEmbeddingDeviceOperation (in0:dram_interleaved) | 11.3 | 7.98 | 4 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 6.1 | 4.30 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 5.1 | 3.63 | 6 |
| SDPAOperation (in0:dram_interleaved) | 3.9 | 2.73 | 2 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 2.0 | 1.44 | 2 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.4 | 0.27 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.3 | 0.20 | 1 |
