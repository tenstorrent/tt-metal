# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_57_46
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

Report: generated/profiler/reports/2026_11_03_22_58_00/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1633.7     | 1210.123 | 1.210  | 1590.0     | 1621.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.9       | 19.210   | 0.019  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1589.1     | 1177.086 | 1.177  | 1172.0     | 1590.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1725.7     | 1278.296 | 1.278  | 1320.0     | 1722.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.1      | 423.062  | 0.423  | 458.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.8      | 273.901  | 0.274  | 363.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1730.6     | 1281.901 | 1.282  | 1697.0     | 1727.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.9       | 14.741   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1609.2     | 1192.025 | 1.192  | 1428.0     | 1533.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1741.6     | 1290.099 | 1.290  | 1573.0     | 1662.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.8      | 424.321  | 0.424  | 404.0      | 576.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.7      | 273.136  | 0.273  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1682.1     | 1246.012 | 1.246  | 1633.7                | 1682.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.9       | 16.975   | 0.017  | 19.9                  | 22.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1599.2     | 1184.556 | 1.185  | 1589.1                | 1599.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1733.7     | 1284.198 | 1.284  | 1725.7                | 1733.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.0      | 423.691  | 0.424  | 571.1                 | 572.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.519  | 0.274  | 368.7                 | 369.2                    |

Final op time: 1.284 us (1733.7 cycles), bucket=TRISC_0 compute

