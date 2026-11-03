# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_15_52
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_16_02/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1634.1     | 1210.420 | 1.210  | 1574.0     | 1639.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.7       | 19.802   | 0.020  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1620.1     | 1200.099 | 1.200  | 1429.0     | 1629.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1713.8     | 1269.506 | 1.270  | 1528.0     | 1721.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 579.8      | 429.481  | 0.429  | 425.0      | 586.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.6      | 278.222  | 0.278  | 369.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1725.5     | 1278.123 | 1.278  | 1686.0     | 1731.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.9       | 15.457   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1613.5     | 1195.185 | 1.195  | 1474.0     | 1614.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1707.7     | 1264.963 | 1.265  | 1567.0     | 1706.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.3      | 420.938  | 0.421  | 414.0      | 578.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.4      | 277.309  | 0.277  | 365.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1679.8     | 1244.272 | 1.244  | 1634.1                | 1679.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.8       | 17.630   | 0.018  | 20.9                  | 23.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1616.8     | 1197.642 | 1.198  | 1613.5                | 1616.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1710.8     | 1267.235 | 1.267  | 1707.7                | 1710.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.0      | 425.210  | 0.425  | 568.3                 | 574.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.0      | 277.765  | 0.278  | 374.4                 | 375.0                    |

Final op time: 1.267 us (1710.8 cycles), bucket=TRISC_0 compute

