# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 22.09 | 7 | 1 | 22.1 |
| layer_time | 50.24 | 19 | 18 | 904.4 |
| layer_time_variate | 63.77 | 26 | 6 | 382.6 |
| head | 0.90 | 8 | 1 | 0.9 |

Estimated full-model device time: **1310.0 ms** over 513 ops.
Measured: device 1307.2 ms (estimate off by +0.2%), steady wall 1445.3 ms, PCC 0.99995 normalized.

Category share of the reduced window: Compute 95.4%, TM 4.6%; matmul weighted FLOPs utilization 14.5%.

| op type | % | device ms | count |
|---|---|---|---|
| MatmulDeviceOperation (in0:dram_interleaved) | 64.6 | 88.51 | 21 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 12.0 | 16.45 | 14 |
| UnaryDeviceOperation (in0:dram_interleaved) | 8.3 | 11.45 | 7 |
| RotaryEmbeddingDeviceOperation (in0:dram_interleaved) | 5.8 | 7.99 | 4 |
| NlpCreateHeadsDeviceOperation (in0:dram_interleaved) | 3.1 | 4.30 | 2 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 2.6 | 3.63 | 6 |
| SDPAOperation (in0:dram_interleaved) | 2.0 | 2.74 | 2 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 1.1 | 1.46 | 2 |
| SliceDeviceOperation (in0:dram_interleaved) | 0.2 | 0.27 | 1 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 0.1 | 0.20 | 1 |
