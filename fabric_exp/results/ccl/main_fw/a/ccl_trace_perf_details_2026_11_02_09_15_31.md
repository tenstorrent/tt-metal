# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_15_31
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_15_41/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1622.1     | 1201.531 | 1.202  | 1573.0     | 1608.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.7       | 19.802   | 0.020  | 18.0       | 23.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1589.7     | 1177.580 | 1.178  | 1334.0     | 1587.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1682.4     | 1246.222 | 1.246  | 1416.0     | 1681.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 579.5      | 429.235  | 0.429  | 442.0      | 591.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.778  | 0.278  | 369.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1691.5     | 1252.938 | 1.253  | 1637.0     | 1680.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.6       | 15.259   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1584.3     | 1173.580 | 1.174  | 1357.0     | 1570.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1679.5     | 1244.074 | 1.244  | 1450.0     | 1666.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.4      | 421.037  | 0.421  | 418.0      | 584.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.7      | 278.272  | 0.278  | 368.0      | 375.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1656.8     | 1227.235 | 1.227  | 1622.1                | 1656.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.7       | 17.531   | 0.018  | 20.6                  | 23.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1587.0     | 1175.580 | 1.176  | 1584.3                | 1587.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1681.0     | 1245.148 | 1.245  | 1679.5                | 1681.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.9      | 425.136  | 0.425  | 568.4                 | 573.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.3      | 278.025  | 0.278  | 375.0                 | 375.3                    |

Final op time: 1.245 us (1681.0 cycles), bucket=TRISC_0 compute

