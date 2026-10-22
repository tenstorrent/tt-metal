# Round summary: paper / accuracy

| section | device ms | ops | x count | contribution ms |
|---|---|---|---|---|
| embed | 9.08 | 6 | 1 | 9.1 |
| layer_time | 18.34 | 13 | 18 | 330.0 |
| layer_time_variate | 23.78 | 18 | 5 | 118.9 |
| layer_last | 10.16 | 24 | 1 | 10.2 |
| head | 0.25 | 5 | 1 | 0.3 |

Estimated full-model device time: **468.4 ms** over 359 ops.
Measured: device 470.2 ms (estimate off by -0.4%), steady wall 540.7 ms, PCC 0.99994 normalized.

Category share of the reduced window: Other 58.1%, Compute 35.7%, TM 6.2%; matmul weighted FLOPs utilization 24.1%.

| op type | % | device ms | count |
|---|---|---|---|
| MinimalMatmulDeviceOperation (in0:dram_interleaved) | 45.0 | 27.71 | 18 |
| FusedRMSNormPostAllGatherDeviceOperation (in0:dram_interleaved) | 13.2 | 8.11 | 9 |
| MatmulDeviceOperation (in0:dram_interleaved) | 12.0 | 7.41 | 9 |
| BinaryNgDeviceOperation (in0:dram_interleaved) | 9.9 | 6.11 | 9 |
| LayerNormDeviceOperation (in0:dram_interleaved) | 7.1 | 4.39 | 9 |
| SDPAOperation (in0:dram_interleaved) | 6.6 | 4.08 | 3 |
| NLPConcatHeadsDeviceOperation (in0:dram_interleaved) | 3.9 | 2.37 | 3 |
| SliceDeviceOperation (in0:dram_interleaved) | 1.3 | 0.82 | 3 |
| ReshapeViewDeviceOperation (in0:dram_interleaved) | 1.0 | 0.61 | 3 |
