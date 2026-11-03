# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_44_48
CCLs: sdpa_reduce_to_all

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 2
Configured compute_block_sizes: 8
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=8, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_45_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5872.0     | 4349.630 | 4.350  | 4480.0     | 5825.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4059.4     | 3006.963 | 3.007  | 3015.0     | 4091.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5617.9     | 4161.400 | 4.161  | 4315.0     | 5532.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6450.1     | 4777.881 | 4.778  | 4948.0     | 6421.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7373.8     | 5462.081 | 5.462  | 5952.0     | 7313.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5662.1     | 4194.119 | 4.194  | 4710.0     | 5597.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5549.2     | 4110.526 | 4.111  | 4566.0     | 5369.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5863.7     | 4343.489 | 4.343  | 5134.0     | 5847.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5026.0     | 3722.993 | 3.723  | 4072.0     | 5011.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5612.2     | 4157.207 | 4.157  | 4889.0     | 5589.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6431.4     | 4764.030 | 4.764  | 5620.0     | 6431.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7355.2     | 5448.267 | 5.448  | 6620.0     | 7341.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6683.8     | 4950.993 | 4.951  | 5253.0     | 6665.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6461.0     | 4785.926 | 4.786  | 5407.0     | 6373.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5835.4     | 4322.548 | 4.323  | 4962.0     | 5843.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5026.1     | 3723.044 | 3.723  | 4098.0     | 5046.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5617.4     | 4161.030 | 4.161  | 4897.0     | 5600.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6411.3     | 4749.133 | 4.749  | 5317.0     | 6412.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7333.9     | 5432.489 | 5.432  | 6397.0     | 7354.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6645.8     | 4922.822 | 4.923  | 4708.0     | 6679.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6456.1     | 4782.333 | 4.782  | 5339.0     | 6357.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5845.2     | 4329.778 | 4.330  | 4421.0     | 5788.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4010.9     | 2971.074 | 2.971  | 2996.0     | 3963.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5587.6     | 4138.970 | 4.139  | 4254.0     | 5522.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6424.6     | 4758.926 | 4.759  | 4823.0     | 6397.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7343.5     | 5439.637 | 5.440  | 5907.0     | 7285.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5637.9     | 4176.259 | 4.176  | 4578.0     | 5544.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5529.5     | 4095.933 | 4.096  | 3745.0     | 5358.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5854.1     | 4336.361 | 4.336  | 5835.4                | 5854.5                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4530.6     | 3356.019 | 3.356  | 4010.9                | 4542.7                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5608.8     | 4154.652 | 4.155  | 5587.6                | 5614.8                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6429.4     | 4762.493 | 4.762  | 6411.3                | 6428.0                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7351.6     | 5445.619 | 5.446  | 7333.9                | 7349.3                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6157.4     | 4561.048 | 4.561  | 5637.9                | 6153.9                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 5999.0     | 4443.680 | 4.444  | 5529.5                | 6002.7                   |

Final op time: 5.446 us (7351.6 cycles), bucket=TRISC_2 compute

