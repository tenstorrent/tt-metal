# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_42_53
CCLs: sdpa_reduce_to_all

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 2
Configured compute_block_sizes: 4
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=4, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_43_08/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5887.8     | 4361.348 | 4.361  | 3382.0     | 5776.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4219.9     | 3125.859 | 3.126  | 3204.0     | 4083.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5650.6     | 4185.593 | 4.186  | 3109.0     | 5566.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6445.6     | 4774.548 | 4.775  | 3981.0     | 6351.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7383.5     | 5469.281 | 5.469  | 4908.0     | 7255.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5759.6     | 4266.400 | 4.266  | 4428.0     | 5639.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5842.8     | 4328.007 | 4.328  | 4868.0     | 5711.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5880.0     | 4355.556 | 4.356  | 4671.0     | 5839.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5050.0     | 3740.763 | 3.741  | 3766.0     | 5155.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5618.7     | 4161.993 | 4.162  | 4377.0     | 5556.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6466.8     | 4790.244 | 4.790  | 5251.0     | 6432.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7374.1     | 5462.304 | 5.462  | 6164.0     | 7337.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6522.1     | 4831.185 | 4.831  | 4238.0     | 6616.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6145.7     | 4552.400 | 4.552  | 4665.0     | 6335.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5834.0     | 4321.504 | 4.322  | 4539.0     | 5785.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5051.7     | 3741.970 | 3.742  | 3700.0     | 5180.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5593.0     | 4142.941 | 4.143  | 4616.0     | 5561.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6427.5     | 4761.133 | 4.761  | 5001.0     | 6393.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7342.3     | 5438.741 | 5.439  | 6092.0     | 7289.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6504.4     | 4818.096 | 4.818  | 4005.0     | 6640.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6149.8     | 4555.385 | 4.555  | 4781.0     | 6354.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5849.9     | 4333.244 | 4.333  | 3179.0     | 5771.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4160.0     | 3081.459 | 3.081  | 3256.0     | 4037.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5610.2     | 4155.696 | 4.156  | 3090.0     | 5556.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6403.8     | 4743.526 | 4.744  | 4073.0     | 6359.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7339.2     | 5436.422 | 5.436  | 4951.0     | 7271.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5744.7     | 4255.363 | 4.255  | 4638.0     | 5598.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5797.3     | 4294.326 | 4.294  | 3981.0     | 5670.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5862.9     | 4342.913 | 4.343  | 5834.0                | 5864.9                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4620.4     | 3422.513 | 3.423  | 4160.0                | 4635.0                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5618.1     | 4161.556 | 4.162  | 5593.0                | 5614.4                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6435.9     | 4767.363 | 4.767  | 6403.8                | 6436.6                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7359.8     | 5451.687 | 5.452  | 7339.2                | 7358.2                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6132.7     | 4542.761 | 4.543  | 5744.7                | 6132.0                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 5983.9     | 4432.530 | 4.433  | 5797.3                | 5994.3                   |

Final op time: 5.452 us (7359.8 cycles), bucket=TRISC_2 compute

