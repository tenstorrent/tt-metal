# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_51_20
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

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=4, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_51_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5855.9     | 4337.674 | 4.338  | 3664.0     | 5934.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3944.3     | 2921.741 | 2.922  | 3234.0     | 3780.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5594.2     | 4143.837 | 4.144  | 3396.0     | 5665.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6446.9     | 4775.481 | 4.775  | 4284.0     | 6504.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7356.2     | 5449.030 | 5.449  | 5175.0     | 7415.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5563.6     | 4121.163 | 4.121  | 4243.0     | 5375.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5529.8     | 4096.148 | 4.096  | 4934.0     | 5315.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5909.6     | 4377.481 | 4.377  | 5119.0     | 5874.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5073.8     | 3758.341 | 3.758  | 3736.0     | 5162.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5636.2     | 4174.993 | 4.175  | 4843.0     | 5597.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6495.2     | 4811.296 | 4.811  | 5730.0     | 6474.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7405.1     | 5485.252 | 5.485  | 6604.0     | 7367.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6692.8     | 4957.615 | 4.958  | 5212.0     | 6754.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6506.7     | 4819.793 | 4.820  | 4716.0     | 6615.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5876.8     | 4353.207 | 4.353  | 4851.0     | 5866.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5065.1     | 3751.889 | 3.752  | 3764.0     | 5111.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5623.0     | 4165.215 | 4.165  | 4945.0     | 5627.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6474.6     | 4796.037 | 4.796  | 5390.0     | 6478.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7381.4     | 5467.689 | 5.468  | 6459.0     | 7369.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6670.2     | 4940.867 | 4.941  | 4284.0     | 6756.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6487.8     | 4805.741 | 4.806  | 4654.0     | 6565.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5827.5     | 4316.652 | 4.317  | 3718.0     | 5915.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3919.8     | 2903.526 | 2.904  | 3281.0     | 3782.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5569.4     | 4125.444 | 4.125  | 3442.0     | 5646.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6420.2     | 4755.711 | 4.756  | 4230.0     | 6492.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7328.8     | 5428.711 | 5.429  | 5209.0     | 7411.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5548.7     | 4110.126 | 4.110  | 4473.0     | 5382.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5496.2     | 4071.230 | 4.071  | 3819.0     | 5307.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5867.4     | 4346.254 | 4.346  | 5827.5                | 5866.3                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4500.7     | 3333.874 | 3.334  | 3919.8                | 4504.7                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5605.7     | 4152.372 | 4.152  | 5569.4                | 5608.6                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6459.3     | 4784.631 | 4.785  | 6420.2                | 6460.8                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7367.9     | 5457.670 | 5.458  | 7328.8                | 7368.8                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6118.8     | 4532.443 | 4.532  | 5548.7                | 6116.9                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6005.1     | 4448.228 | 4.448  | 5496.2                | 6008.8                   |

Final op time: 5.458 us (7367.9 cycles), bucket=TRISC_2 compute

