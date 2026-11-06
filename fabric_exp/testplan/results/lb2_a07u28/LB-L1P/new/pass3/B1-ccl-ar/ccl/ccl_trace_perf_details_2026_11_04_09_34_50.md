# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_34_50
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

Report: generated/profiler/reports/2026_11_04_09_35_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1881.4     | 1393.605 | 1.394  | 1836.0     | 1873.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1744.5     | 1292.222 | 1.292  | 1673.0     | 1740.0        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1752.2     | 1297.901 | 1.298  | 1618.0     | 1750.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1842.9     | 1365.136 | 1.365  | 1697.0     | 1849.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.8      | 426.543  | 0.427  | 481.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.8      | 273.185  | 0.273  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1788.2     | 1324.617 | 1.325  | 1713.0     | 1782.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1769.8     | 1310.938 | 1.311  | 1678.0     | 1771.5        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1791.8     | 1327.284 | 1.327  | 1679.0     | 1771.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1885.9     | 1396.963 | 1.397  | 1766.0     | 1867.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.4      | 425.506  | 0.426  | 469.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 364.0      | 369.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1834.8     | 1359.111 | 1.359  | 1788.2                | 1834.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1757.1     | 1301.580 | 1.302  | 1744.5                | 1757.1                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1772.0     | 1312.593 | 1.313  | 1752.2                | 1772.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1864.4     | 1381.049 | 1.381  | 1842.9                | 1864.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 575.1      | 426.025  | 0.426  | 574.4                 | 575.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.0      | 273.358  | 0.273  | 368.8                 | 369.0                    |

Final op time: 1.381 us (1864.4 cycles), bucket=TRISC_0 compute

