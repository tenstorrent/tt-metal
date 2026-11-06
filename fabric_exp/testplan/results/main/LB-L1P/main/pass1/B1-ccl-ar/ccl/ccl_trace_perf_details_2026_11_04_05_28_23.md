# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_28_23
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

Report: generated/profiler/reports/2026_11_04_05_28_33/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1778.9     | 1317.704 | 1.318  | 1718.0     | 1781.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1769.9     | 1311.012 | 1.311  | 1687.0     | 1767.0        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1741.2     | 1289.753 | 1.290  | 1319.0     | 1723.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1837.8     | 1361.309 | 1.361  | 1404.0     | 1817.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.2      | 423.111  | 0.423  | 457.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.2      | 274.198  | 0.274  | 364.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1873.0     | 1387.432 | 1.387  | 1794.0     | 1844.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1752.4     | 1298.049 | 1.298  | 1691.0     | 1740.5        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1758.5     | 1302.568 | 1.303  | 1542.0     | 1710.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1847.9     | 1368.790 | 1.369  | 1681.0     | 1800.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.3      | 423.185  | 0.423  | 457.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.259  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1826.0     | 1352.568 | 1.353  | 1778.9                | 1826.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1761.1     | 1304.531 | 1.305  | 1752.4                | 1761.1                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1749.8     | 1296.160 | 1.296  | 1741.2                | 1749.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1842.8     | 1365.049 | 1.365  | 1837.8                | 1842.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.2      | 423.148  | 0.423  | 571.2                 | 571.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.728  | 0.274  | 368.9                 | 369.5                    |

Final op time: 1.365 us (1842.8 cycles), bucket=TRISC_0 compute

