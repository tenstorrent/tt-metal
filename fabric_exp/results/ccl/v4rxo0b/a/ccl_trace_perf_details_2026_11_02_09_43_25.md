# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_43_25
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

Report: generated/profiler/reports/2026_11_02_09_43_41/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5970.4     | 4422.556 | 4.423  | 4972.0     | 5971.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3727.6     | 2761.193 | 2.761  | 3101.0     | 3527.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5710.5     | 4229.978 | 4.230  | 4711.0     | 5697.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6560.2     | 4859.378 | 4.859  | 5578.0     | 6560.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7473.6     | 5536.037 | 5.536  | 6495.0     | 7460.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5273.0     | 3905.948 | 3.906  | 4266.0     | 5118.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5210.5     | 3859.630 | 3.860  | 4613.0     | 5002.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5846.2     | 4330.533 | 4.331  | 3804.0     | 5873.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5331.3     | 3949.111 | 3.949  | 3880.0     | 5446.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5593.9     | 4143.652 | 4.144  | 3501.0     | 5635.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6391.6     | 4734.481 | 4.734  | 4419.0     | 6362.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7334.3     | 5432.807 | 5.433  | 5293.0     | 7325.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7017.9     | 5198.422 | 5.198  | 5111.0     | 7162.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6904.1     | 5114.126 | 5.114  | 5212.0     | 6842.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5773.4     | 4276.563 | 4.277  | 3705.0     | 5772.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5316.2     | 3937.896 | 3.938  | 3829.0     | 5411.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5572.5     | 4127.800 | 4.128  | 3510.0     | 5604.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6314.8     | 4677.659 | 4.678  | 4264.0     | 6263.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7265.1     | 5381.585 | 5.382  | 5340.0     | 7250.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6975.3     | 5166.896 | 5.167  | 3597.0     | 7121.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6890.9     | 5104.385 | 5.104  | 5016.0     | 6834.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5942.8     | 4402.089 | 4.402  | 4927.0     | 5950.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3692.4     | 2735.133 | 2.735  | 2902.0     | 3522.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5685.5     | 4211.481 | 4.211  | 4644.0     | 5702.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6529.6     | 4836.770 | 4.837  | 5515.0     | 6555.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7440.7     | 5511.644 | 5.512  | 6430.0     | 7460.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5269.4     | 3903.259 | 3.903  | 4671.0     | 5128.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5174.8     | 3833.215 | 3.833  | 3420.0     | 5037.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5883.2     | 4357.935 | 4.358  | 5773.4                | 5894.5                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4516.9     | 3345.833 | 3.346  | 3692.4                | 4521.9                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5640.6     | 4178.228 | 4.178  | 5572.5                | 5639.7                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6449.0     | 4777.072 | 4.777  | 6314.8                | 6460.6                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7378.5     | 5465.519 | 5.466  | 7265.1                | 7387.5                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6133.9     | 4543.631 | 4.544  | 5269.4                | 6124.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6045.1     | 4477.839 | 4.478  | 5174.8                | 6050.7                   |

Final op time: 5.466 us (7378.5 cycles), bucket=TRISC_2 compute

