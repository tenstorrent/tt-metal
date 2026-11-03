# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_54_47
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_55_00/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1625.7     | 1204.247 | 1.204  | 1573.0     | 1612.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 23.9       | 17.728   | 0.018  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1593.5     | 1180.370 | 1.180  | 1499.0     | 1594.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1688.4     | 1250.667 | 1.251  | 1599.0     | 1683.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.3      | 426.148  | 0.426  | 400.0      | 581.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.7      | 278.272  | 0.278  | 371.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1692.0     | 1253.333 | 1.253  | 1651.0     | 1689.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.9       | 15.457   | 0.015  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1582.1     | 1171.951 | 1.172  | 1447.0     | 1562.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1677.9     | 1242.889 | 1.243  | 1544.0     | 1660.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.7      | 421.235  | 0.421  | 418.0      | 580.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.728  | 0.278  | 368.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1658.9     | 1228.790 | 1.229  | 1625.7                | 1658.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.4       | 16.593   | 0.017  | 20.9                  | 22.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1587.8     | 1176.160 | 1.176  | 1582.1                | 1587.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1683.2     | 1246.778 | 1.247  | 1677.9                | 1683.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.0      | 423.691  | 0.424  | 568.7                 | 572.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.3      | 278.000  | 0.278  | 374.9                 | 375.3                    |

Final op time: 1.247 us (1683.2 cycles), bucket=TRISC_0 compute

