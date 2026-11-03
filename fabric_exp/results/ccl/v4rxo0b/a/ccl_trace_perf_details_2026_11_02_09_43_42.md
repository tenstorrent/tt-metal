# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_43_42
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

Report: generated/profiler/reports/2026_11_02_09_43_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5872.2     | 4349.815 | 4.350  | 5579.0     | 5842.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4463.9     | 3306.570 | 3.307  | 4132.0     | 4430.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5636.6     | 4175.252 | 4.175  | 5304.0     | 5605.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6445.5     | 4774.459 | 4.774  | 6171.0     | 6411.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7365.9     | 5456.244 | 5.456  | 7097.0     | 7329.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6548.9     | 4851.015 | 4.851  | 5224.0     | 6573.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6552.7     | 4853.822 | 4.854  | 6071.0     | 6571.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5897.1     | 4368.200 | 4.368  | 5562.0     | 5846.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4489.5     | 3325.556 | 3.326  | 4139.0     | 4434.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5678.4     | 4206.237 | 4.206  | 5284.0     | 5610.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6470.9     | 4793.281 | 4.793  | 6147.0     | 6424.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7386.2     | 5471.244 | 5.471  | 7056.0     | 7331.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6571.1     | 4867.444 | 4.867  | 6202.0     | 6566.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6524.2     | 4832.770 | 4.833  | 6160.0     | 6501.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5853.3     | 4335.756 | 4.336  | 5552.0     | 5819.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4479.0     | 3317.756 | 3.318  | 4126.0     | 4450.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5645.9     | 4182.163 | 4.182  | 5324.0     | 5604.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6443.1     | 4772.644 | 4.773  | 6146.0     | 6414.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7353.5     | 5447.030 | 5.447  | 7070.0     | 7325.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6551.7     | 4853.133 | 4.853  | 5673.0     | 6567.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6519.9     | 4829.519 | 4.830  | 5510.0     | 6529.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5866.6     | 4345.622 | 4.346  | 5585.0     | 5839.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4448.2     | 3294.963 | 3.295  | 4096.0     | 4416.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5633.8     | 4173.156 | 4.173  | 5356.0     | 5604.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6441.6     | 4771.519 | 4.772  | 6151.0     | 6419.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7359.2     | 5451.237 | 5.451  | 7096.0     | 7339.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6532.2     | 4838.644 | 4.839  | 5447.0     | 6550.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6516.2     | 4826.785 | 4.827  | 5172.0     | 6523.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5872.3     | 4349.848 | 4.350  | 5853.3                | 5869.4                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4470.1     | 3311.211 | 3.311  | 4448.2                | 4471.4                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5648.7     | 4184.202 | 4.184  | 5633.8                | 5641.3                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6450.3     | 4777.976 | 4.778  | 6441.6                | 6444.3                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7366.2     | 5456.439 | 5.456  | 7353.5                | 7362.6                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6551.0     | 4852.559 | 4.853  | 6532.2                | 6550.3                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6528.2     | 4835.724 | 4.836  | 6516.2                | 6522.0                   |

Final op time: 5.456 us (7366.2 cycles), bucket=TRISC_2 compute

