# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_20_58
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

Report: generated/profiler/reports/2026_11_02_09_21_16/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1627.5     | 1205.556 | 1.206  | 1575.0     | 1613.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.4       | 20.272   | 0.020  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1564.1     | 1158.593 | 1.159  | 1416.0     | 1563.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1694.8     | 1255.432 | 1.255  | 1538.0     | 1686.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.2      | 420.148  | 0.420  | 453.0      | 570.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.704  | 0.278  | 370.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1695.6     | 1256.000 | 1.256  | 1653.0     | 1692.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.1       | 14.914   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1553.8     | 1150.938 | 1.151  | 1351.0     | 1538.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1689.8     | 1251.728 | 1.252  | 1490.0     | 1673.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 561.5      | 415.926  | 0.416  | 405.0      | 566.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.4      | 278.074  | 0.278  | 370.0      | 375.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1661.5     | 1230.778 | 1.231  | 1627.5                | 1661.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.8       | 17.593   | 0.018  | 20.1                  | 23.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1558.9     | 1154.765 | 1.155  | 1553.8                | 1558.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1692.3     | 1253.580 | 1.254  | 1689.8                | 1692.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 564.4      | 418.037  | 0.418  | 561.5                 | 564.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.1      | 277.889  | 0.278  | 374.9                 | 375.1                    |

Final op time: 1.254 us (1692.3 cycles), bucket=TRISC_0 compute

