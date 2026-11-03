# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_47_06
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_47_18/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1633.3     | 1209.877 | 1.210  | 1576.0     | 1624.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.1       | 21.580   | 0.022  | 18.0       | 26.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1566.2     | 1160.148 | 1.160  | 1414.0     | 1570.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1692.2     | 1253.481 | 1.253  | 1541.0     | 1694.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.4      | 423.284  | 0.423  | 438.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 373.9      | 276.988  | 0.277  | 371.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1695.5     | 1255.926 | 1.256  | 1652.0     | 1690.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.0       | 15.556   | 0.016  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1554.2     | 1151.235 | 1.151  | 1385.0     | 1535.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1689.0     | 1251.136 | 1.251  | 1519.0     | 1660.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 564.8      | 418.395  | 0.418  | 443.0      | 566.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.3      | 278.000  | 0.278  | 372.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1664.4     | 1232.901 | 1.233  | 1633.3                | 1664.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.1       | 18.568   | 0.019  | 21.0                  | 25.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1560.2     | 1155.691 | 1.156  | 1554.2                | 1560.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1690.6     | 1252.309 | 1.252  | 1689.0                | 1690.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 568.1      | 420.840  | 0.421  | 564.8                 | 568.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.6      | 277.494  | 0.277  | 373.9                 | 374.6                    |

Final op time: 1.252 us (1690.6 cycles), bucket=TRISC_0 compute

