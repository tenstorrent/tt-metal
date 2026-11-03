# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_01_26
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_10_01_36/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1636.0     | 1211.877 | 1.212  | 1577.0     | 1637.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.3       | 21.704   | 0.022  | 18.0       | 26.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1644.6     | 1218.198 | 1.218  | 1483.0     | 1632.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1739.4     | 1288.420 | 1.288  | 1586.0     | 1726.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.3      | 427.654  | 0.428  | 419.0      | 587.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.4      | 277.333  | 0.277  | 369.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1745.1     | 1292.642 | 1.293  | 1660.0     | 1732.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.0       | 14.815   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1625.8     | 1204.321 | 1.204  | 1495.0     | 1618.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1723.8     | 1276.889 | 1.277  | 1587.0     | 1714.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.6      | 420.420  | 0.420  | 447.0      | 570.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.6      | 278.222  | 0.278  | 370.0      | 376.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1690.5     | 1252.259 | 1.252  | 1636.0                | 1690.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.6       | 18.259   | 0.018  | 20.0                  | 24.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1635.2     | 1211.259 | 1.211  | 1625.8                | 1635.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1731.6     | 1282.654 | 1.283  | 1723.8                | 1731.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.5      | 424.037  | 0.424  | 567.6                 | 572.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.0      | 277.778  | 0.278  | 374.4                 | 375.0                    |

Final op time: 1.283 us (1731.6 cycles), bucket=TRISC_0 compute

