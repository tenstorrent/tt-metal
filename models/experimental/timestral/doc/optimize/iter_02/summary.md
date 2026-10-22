# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 9.29 | 6 | 1 | 9.3 |
| layer_time | 20.19 | 12 | 18 | 363.4 |
| layer_time_variate | 25.73 | 17 | 6 | 154.4 |
| head | 0.73 | 7 | 1 | 0.7 |

Estimated full-model device time: **527.9 ms** over 331 ops.
Measured: device 530.9 ms (estimate off by -0.6%), steady wall 679.6 ms, PCC 0.99993 normalized.

Category share of the reduced window: Compute 56.4%, Other 35.1%, TM 8.6%; matmul weighted FLOPs utilization 32.5%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 35.1 | 19.63 | 10 |
| MatmulDeviceOperation (in0:dram_interleaved) | 22.3 | 12.49 | 9 |
| RotaryEmbeddingLlamaDeviceOperation (in0:dram_interleaved) | 11.6 | 6.50 | 4 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 10.8 | 6.04 | 7 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 7.7 | 4.31 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 6.5 | 3.63 | 6 |
| SDPAOperation (in0:dram_interleaved) | 5.1 | 2.88 | 2 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.5 | 0.27 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.4 | 0.21 | 1 |
