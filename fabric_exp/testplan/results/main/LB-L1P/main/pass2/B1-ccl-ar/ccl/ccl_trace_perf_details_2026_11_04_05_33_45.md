# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_33_45
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_04_05_33_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1772.7     | 1313.086 | 1.313  | 1730.0     | 1763.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1767.2     | 1309.012 | 1.309  | 1687.0     | 1767.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1761.4     | 1304.765 | 1.305  | 1459.0     | 1776.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1854.5     | 1373.704 | 1.374  | 1554.0     | 1876.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.8      | 425.802  | 0.426  | 454.0      | 577.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.284  | 0.273  | 363.0      | 368.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1885.8     | 1396.914 | 1.397  | 1801.0     | 1888.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1741.7     | 1290.148 | 1.290  | 1641.0     | 1719.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1758.2     | 1302.395 | 1.302  | 1554.0     | 1728.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1848.8     | 1369.506 | 1.370  | 1646.0     | 1814.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.4      | 423.975  | 0.424  | 427.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.8      | 273.185  | 0.273  | 358.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1829.2     | 1355.000 | 1.355  | 1772.7                | 1829.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1754.4     | 1299.580 | 1.300  | 1741.7                | 1754.4                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1759.8     | 1303.580 | 1.304  | 1758.2                | 1759.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1851.7     | 1371.605 | 1.372  | 1848.8                | 1851.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.6      | 424.889  | 0.425  | 572.4                 | 573.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.9      | 273.235  | 0.273  | 368.8                 | 368.9                    |

Final op time: 1.372 us (1851.7 cycles), bucket=TRISC_0 compute

