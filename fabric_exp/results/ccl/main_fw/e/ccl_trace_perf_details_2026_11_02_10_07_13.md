# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_07_13
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

Report: generated/profiler/reports/2026_11_02_10_07_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7166.8     | 5308.741 | 5.309  | 6576.0     | 7153.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5157.6     | 3820.481 | 3.820  | 4397.0     | 5315.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6965.1     | 5159.333 | 5.159  | 6546.0     | 6939.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7557.2     | 5597.904 | 5.598  | 6902.0     | 7547.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8626.8     | 6390.207 | 6.390  | 7994.0     | 8631.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6759.3     | 5006.919 | 5.007  | 4962.0     | 6760.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6785.0     | 5025.896 | 5.026  | 6504.0     | 6747.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6210.5     | 4600.385 | 4.600  | 5808.0     | 6140.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5414.6     | 4010.785 | 4.011  | 4537.0     | 5419.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5973.1     | 4424.533 | 4.425  | 5606.0     | 5884.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6806.7     | 5042.030 | 5.042  | 6421.0     | 6739.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7709.2     | 5710.504 | 5.711  | 7299.0     | 7640.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7004.6     | 5188.556 | 5.189  | 6475.0     | 6999.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6846.4     | 5071.422 | 5.071  | 6319.0     | 6813.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6202.9     | 4594.778 | 4.595  | 5738.0     | 6124.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5366.0     | 3974.837 | 3.975  | 4432.0     | 5532.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5971.6     | 4423.400 | 4.423  | 5649.0     | 5887.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6799.7     | 5036.785 | 5.037  | 6303.0     | 6732.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7706.8     | 5708.733 | 5.709  | 7217.0     | 7639.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6946.8     | 5145.756 | 5.146  | 5143.0     | 6953.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6818.6     | 5050.793 | 5.051  | 6434.0     | 6794.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7164.6     | 5307.081 | 5.307  | 6374.0     | 7175.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5124.8     | 3796.156 | 3.796  | 4399.0     | 5171.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6976.9     | 5168.059 | 5.168  | 6435.0     | 6972.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7544.2     | 5588.267 | 5.588  | 6716.0     | 7562.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8610.8     | 6378.393 | 6.378  | 7817.0     | 8636.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6743.0     | 4994.785 | 4.995  | 5938.0     | 6736.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6729.1     | 4984.496 | 4.984  | 3748.0     | 6749.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6686.2     | 4952.746 | 4.953  | 6202.9                | 6687.5                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 5265.8     | 3900.565 | 3.901  | 5124.8                | 5261.8                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 6471.7     | 4793.831 | 4.794  | 5971.6                | 6469.1                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 7176.9     | 5316.246 | 5.316  | 6799.7                | 7175.4                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 8163.4     | 6046.959 | 6.047  | 7706.8                | 8160.0                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6863.4     | 5084.004 | 5.084  | 6743.0                | 6853.1                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6794.8     | 5033.152 | 5.033  | 6729.1                | 6801.8                   |

Final op time: 6.047 us (8163.4 cycles), bucket=TRISC_2 compute

