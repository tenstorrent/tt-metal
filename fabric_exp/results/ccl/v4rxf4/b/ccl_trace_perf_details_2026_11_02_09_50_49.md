# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_50_49
CCLs: sdpa_reduce_to_all

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 4
Configured compute_block_sizes: 8
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_l_chunks=4, tiles_per_l_chunk=4, block_size=8, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_51_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6540.8     | 4845.059 | 4.845  | 4959.0     | 6757.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4845.2     | 3589.059 | 3.589  | 3755.0     | 4913.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6325.4     | 4685.511 | 4.686  | 4669.0     | 6476.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6991.6     | 5179.000 | 5.179  | 5562.0     | 7117.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8007.5     | 5931.467 | 5.931  | 6476.0     | 8200.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6265.2     | 4640.919 | 4.641  | 5268.0     | 6318.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6282.1     | 4653.393 | 4.653  | 5293.0     | 6174.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5863.1     | 4343.030 | 4.343  | 5301.0     | 5731.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5215.2     | 3863.148 | 3.863  | 4133.0     | 5529.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5605.5     | 4152.200 | 4.152  | 5140.0     | 5478.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6456.9     | 4782.859 | 4.783  | 5928.0     | 6319.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7365.1     | 5455.659 | 5.456  | 6902.0     | 7226.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6547.4     | 4849.889 | 4.850  | 5482.0     | 6820.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6317.8     | 4679.844 | 4.680  | 5684.0     | 6277.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5833.6     | 4321.222 | 4.321  | 5212.0     | 5715.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5212.8     | 3861.319 | 3.861  | 4233.0     | 5522.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5579.2     | 4132.711 | 4.133  | 5081.0     | 5468.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6445.1     | 4774.148 | 4.774  | 5872.0     | 6327.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7347.9     | 5442.911 | 5.443  | 6759.0     | 7227.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6538.4     | 4843.281 | 4.843  | 5247.0     | 6841.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6326.2     | 4686.111 | 4.686  | 5541.0     | 6274.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6534.3     | 4840.215 | 4.840  | 5006.0     | 6663.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4818.7     | 3569.437 | 3.569  | 3605.0     | 4903.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6322.1     | 4683.030 | 4.683  | 4733.0     | 6499.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6975.5     | 5167.059 | 5.167  | 5480.0     | 6989.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7986.6     | 5915.970 | 5.916  | 6483.0     | 8068.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6241.7     | 4623.467 | 4.623  | 5334.0     | 6321.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6221.1     | 4608.200 | 4.608  | 3791.0     | 6104.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6193.0     | 4587.381 | 4.587  | 5833.6                | 6198.7                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 5023.0     | 3720.741 | 3.721  | 4818.7                | 5029.0                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5958.0     | 4413.363 | 4.413  | 5579.2                | 5963.8                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6717.3     | 4975.767 | 4.976  | 6445.1                | 6716.2                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7676.8     | 5686.502 | 5.687  | 7347.9                | 7675.9                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6398.2     | 4739.389 | 4.739  | 6241.7                | 6401.8                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6286.8     | 4656.887 | 4.657  | 6221.1                | 6299.9                   |

Final op time: 5.687 us (7676.8 cycles), bucket=TRISC_2 compute

