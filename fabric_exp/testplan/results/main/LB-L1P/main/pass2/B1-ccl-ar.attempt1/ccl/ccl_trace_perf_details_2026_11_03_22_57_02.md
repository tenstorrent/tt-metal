# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_57_02
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_03_22_57_16/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1894.5     | 1403.333 | 1.403  | 1850.0     | 1882.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.5       | 21.086   | 0.021  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1865.9     | 1382.123 | 1.382  | 1385.0     | 1889.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1992.0     | 1475.531 | 1.476  | 1523.0     | 2019.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.5      | 422.568  | 0.423  | 403.0      | 570.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.8      | 273.160  | 0.273  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 2004.9     | 1485.136 | 1.485  | 1905.0     | 2017.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.0       | 14.815   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1901.7     | 1408.691 | 1.409  | 1638.0     | 1816.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 2034.4     | 1506.988 | 1.507  | 1785.0     | 1950.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.5      | 422.617  | 0.423  | 422.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1949.7     | 1444.235 | 1.444  | 1894.5                | 1949.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.2       | 17.951   | 0.018  | 20.0                  | 24.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1883.8     | 1395.407 | 1.395  | 1865.9                | 1883.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 2013.2     | 1491.259 | 1.491  | 1992.0                | 2013.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.5      | 422.593  | 0.423  | 570.5                 | 570.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.0      | 273.346  | 0.273  | 368.8                 | 369.0                    |

Final op time: 1.491 us (2013.2 cycles), bucket=TRISC_0 compute

