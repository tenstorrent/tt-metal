# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_42_24
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

Report: generated/profiler/reports/2026_11_02_09_42_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7447.4     | 5516.570 | 5.517  | 6948.0     | 7406.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5361.6     | 3971.526 | 3.972  | 4330.0     | 5237.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 7357.7     | 5450.148 | 5.450  | 6698.0     | 7314.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7812.8     | 5787.259 | 5.787  | 7287.0     | 7774.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8890.7     | 6585.733 | 6.586  | 8360.0     | 8852.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6923.2     | 5128.296 | 5.128  | 5972.0     | 6936.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6950.2     | 5148.281 | 5.148  | 6383.0     | 6925.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6650.7     | 4926.422 | 4.926  | 6144.0     | 6605.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5339.9     | 3955.467 | 3.955  | 4227.0     | 5341.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6396.6     | 4738.222 | 4.738  | 5889.0     | 6362.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7243.8     | 5365.793 | 5.366  | 6723.0     | 7202.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8151.1     | 6037.852 | 6.038  | 7635.0     | 8100.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6929.0     | 5132.622 | 5.133  | 6117.0     | 6934.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6889.1     | 5103.037 | 5.103  | 6297.0     | 6894.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6624.7     | 4907.200 | 4.907  | 6199.0     | 6576.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5334.6     | 3951.526 | 3.952  | 4420.0     | 5331.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6390.4     | 4733.622 | 4.734  | 5920.0     | 6331.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7229.4     | 5355.089 | 5.355  | 6808.0     | 7183.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8136.1     | 6026.726 | 6.027  | 7745.0     | 8090.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6901.2     | 5112.000 | 5.112  | 5311.0     | 6922.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6831.9     | 5060.689 | 5.061  | 6276.0     | 6830.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7444.1     | 5514.126 | 5.514  | 6775.0     | 7448.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5344.2     | 3958.667 | 3.959  | 4506.0     | 5475.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 7352.9     | 5446.622 | 5.447  | 6685.0     | 7343.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7799.5     | 5777.430 | 5.777  | 7101.0     | 7808.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8878.7     | 6576.837 | 6.577  | 8202.0     | 8885.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6901.2     | 5112.000 | 5.112  | 5851.0     | 6914.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6874.7     | 5092.370 | 5.092  | 4846.0     | 6894.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 7041.7     | 5216.080 | 5.216  | 6624.7                | 7047.4                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 5345.1     | 3959.296 | 3.959  | 5334.6                | 5342.0                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 6874.4     | 5092.154 | 5.092  | 6390.4                | 6874.8                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 7521.4     | 5571.393 | 5.571  | 7229.4                | 7521.7                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 8514.2     | 6306.787 | 6.307  | 8136.1                | 8514.9                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6913.7     | 5121.230 | 5.121  | 6901.2                | 6912.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6886.5     | 5101.094 | 5.101  | 6831.9                | 6881.9                   |

Final op time: 6.307 us (8514.2 cycles), bucket=TRISC_2 compute

