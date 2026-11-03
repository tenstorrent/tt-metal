# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_55_00
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_55_13/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1621.3     | 1200.938 | 1.201  | 1575.0     | 1612.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 23.1       | 17.111   | 0.017  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1593.8     | 1180.617 | 1.181  | 1425.0     | 1588.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1689.7     | 1251.630 | 1.252  | 1519.0     | 1687.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 579.8      | 429.481  | 0.429  | 432.0      | 589.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.654  | 0.278  | 369.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1699.1     | 1258.593 | 1.259  | 1653.0     | 1694.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.0       | 14.815   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1588.1     | 1176.370 | 1.176  | 1453.0     | 1569.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1682.7     | 1246.469 | 1.246  | 1547.0     | 1674.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.4      | 420.321  | 0.420  | 418.0      | 577.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.654  | 0.278  | 365.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1660.2     | 1229.765 | 1.230  | 1621.3                | 1660.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.6       | 15.963   | 0.016  | 20.0                  | 21.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1591.0     | 1178.494 | 1.178  | 1588.1                | 1591.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1686.2     | 1249.049 | 1.249  | 1682.7                | 1686.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.6      | 424.901  | 0.425  | 567.4                 | 573.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.654  | 0.278  | 374.8                 | 374.8                    |

Final op time: 1.249 us (1686.2 cycles), bucket=TRISC_0 compute

