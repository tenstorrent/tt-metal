# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_33_26
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

Report: generated/profiler/reports/2026_11_04_09_33_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1788.5     | 1324.840 | 1.325  | 1717.0     | 1788.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1761.9     | 1305.111 | 1.305  | 1690.0     | 1767.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1758.8     | 1302.790 | 1.303  | 1668.0     | 1753.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1850.6     | 1370.815 | 1.371  | 1768.0     | 1841.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.6      | 424.889  | 0.425  | 402.0      | 577.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.704  | 0.274  | 359.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1866.7     | 1382.765 | 1.383  | 1780.0     | 1854.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1734.8     | 1285.062 | 1.285  | 1690.0     | 1734.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1748.0     | 1294.840 | 1.295  | 1636.0     | 1732.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1838.8     | 1362.074 | 1.362  | 1725.0     | 1828.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.9      | 424.395  | 0.424  | 446.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.7      | 273.136  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1827.6     | 1353.802 | 1.354  | 1788.5                | 1827.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1748.4     | 1295.086 | 1.295  | 1734.8                | 1748.4                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1753.4     | 1298.815 | 1.299  | 1748.0                | 1753.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1844.7     | 1366.444 | 1.366  | 1838.8                | 1844.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.3      | 424.642  | 0.425  | 572.9                 | 573.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.1      | 273.420  | 0.273  | 368.7                 | 369.1                    |

Final op time: 1.366 us (1844.7 cycles), bucket=TRISC_0 compute

