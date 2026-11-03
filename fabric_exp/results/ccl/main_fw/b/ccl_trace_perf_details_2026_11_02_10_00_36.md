# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_00_36
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_10_00_46/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1641.7     | 1216.099 | 1.216  | 1575.0     | 1640.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.6       | 21.926   | 0.022  | 18.0       | 27.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1607.1     | 1190.469 | 1.190  | 1459.0     | 1598.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1742.1     | 1290.444 | 1.290  | 1586.0     | 1737.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.2      | 420.889  | 0.421  | 427.0      | 571.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.704  | 0.278  | 371.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1730.1     | 1281.580 | 1.282  | 1701.0     | 1730.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.8       | 15.383   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1588.6     | 1176.741 | 1.177  | 1401.0     | 1595.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1724.1     | 1277.111 | 1.277  | 1540.0     | 1728.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 561.3      | 415.778  | 0.416  | 438.0      | 564.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.7      | 277.531  | 0.278  | 372.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1685.9     | 1248.840 | 1.249  | 1641.7                | 1685.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.2       | 18.654   | 0.019  | 20.8                  | 25.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1597.9     | 1183.605 | 1.184  | 1588.6                | 1597.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1733.1     | 1283.778 | 1.284  | 1724.1                | 1733.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 564.8      | 418.333  | 0.418  | 561.3                 | 564.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.617  | 0.278  | 374.7                 | 374.8                    |

Final op time: 1.284 us (1733.1 cycles), bucket=TRISC_0 compute

