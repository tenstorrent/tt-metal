# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_51_52
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

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=8, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_52_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5913.1     | 4380.111 | 4.380  | 5596.0     | 5889.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4488.5     | 3324.793 | 3.325  | 4092.0     | 4438.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5695.9     | 4219.156 | 4.219  | 5325.0     | 5637.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6490.1     | 4807.481 | 4.807  | 6152.0     | 6459.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7409.0     | 5488.163 | 5.488  | 7100.0     | 7379.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6545.0     | 4848.133 | 4.848  | 5170.0     | 6572.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6565.7     | 4863.452 | 4.863  | 6003.0     | 6570.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5912.9     | 4379.889 | 4.380  | 5650.0     | 5835.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4503.0     | 3335.556 | 3.336  | 4110.0     | 4427.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5707.3     | 4227.607 | 4.228  | 5402.0     | 5641.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6483.4     | 4802.526 | 4.803  | 6191.0     | 6418.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7402.2     | 5483.148 | 5.483  | 7139.0     | 7332.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6557.9     | 4857.674 | 4.858  | 5807.0     | 6546.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6509.2     | 4821.644 | 4.822  | 6082.0     | 6496.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5918.5     | 4384.067 | 4.384  | 5573.0     | 5858.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4520.5     | 3348.533 | 3.349  | 4156.0     | 4472.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5727.8     | 4242.837 | 4.243  | 5299.0     | 5685.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6513.3     | 4824.644 | 4.825  | 6111.0     | 6464.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7420.5     | 5496.696 | 5.497  | 7022.0     | 7368.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6560.1     | 4859.296 | 4.859  | 4900.0     | 6594.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6492.4     | 4809.156 | 4.809  | 5984.0     | 6526.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5899.5     | 4370.000 | 4.370  | 5641.0     | 5862.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4486.3     | 3323.163 | 3.323  | 4086.0     | 4452.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5665.9     | 4196.970 | 4.197  | 5337.0     | 5618.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6472.1     | 4794.170 | 4.794  | 6184.0     | 6446.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7394.9     | 5477.667 | 5.478  | 7097.0     | 7361.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6526.8     | 4834.674 | 4.835  | 5826.0     | 6534.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6492.5     | 4809.289 | 4.809  | 4902.0     | 6542.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5911.0     | 4378.517 | 4.379  | 5899.5                | 5913.0                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4499.6     | 3333.011 | 3.333  | 4486.3                | 4495.7                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5699.2     | 4221.643 | 4.222  | 5665.9                | 5701.6                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6489.7     | 4807.206 | 4.807  | 6472.1                | 6486.8                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7406.7     | 5486.419 | 5.486  | 7394.9                | 7405.6                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6547.4     | 4849.944 | 4.850  | 6526.8                | 6551.4                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6514.9     | 4825.885 | 4.826  | 6492.4                | 6500.9                   |

Final op time: 5.486 us (7406.7 cycles), bucket=TRISC_2 compute

