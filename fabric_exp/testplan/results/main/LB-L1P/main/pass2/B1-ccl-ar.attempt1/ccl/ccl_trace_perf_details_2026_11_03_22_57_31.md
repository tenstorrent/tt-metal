# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_57_31
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_03_22_57_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1641.7     | 1216.074 | 1.216  | 1595.0     | 1621.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.9       | 19.951   | 0.020  | 18.0       | 19.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1599.7     | 1184.938 | 1.185  | 1203.0     | 1614.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1734.6     | 1284.889 | 1.285  | 1348.0     | 1750.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 566.0      | 419.259  | 0.419  | 407.0      | 571.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.481  | 0.273  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1741.5     | 1289.975 | 1.290  | 1685.0     | 1734.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.420   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1601.8     | 1186.519 | 1.187  | 1449.0     | 1541.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1737.8     | 1287.259 | 1.287  | 1593.0     | 1668.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.4      | 423.284  | 0.423  | 487.0      | 573.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 370.3      | 274.272  | 0.274  | 365.0      | 370.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1691.6     | 1253.025 | 1.253  | 1641.7                | 1691.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.2       | 17.185   | 0.017  | 19.5                  | 23.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1600.7     | 1185.728 | 1.186  | 1599.7                | 1600.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1736.2     | 1286.074 | 1.286  | 1734.6                | 1736.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 568.7      | 421.272  | 0.421  | 566.0                 | 568.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.877  | 0.274  | 369.2                 | 369.7                    |

Final op time: 1.286 us (1736.2 cycles), bucket=TRISC_0 compute

