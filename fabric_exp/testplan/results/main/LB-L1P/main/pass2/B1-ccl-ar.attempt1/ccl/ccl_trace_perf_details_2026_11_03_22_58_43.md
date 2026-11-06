# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_58_43
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_03_22_58_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1640.9     | 1215.506 | 1.216  | 1593.0     | 1627.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.0       | 17.753   | 0.018  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1622.6     | 1201.901 | 1.202  | 1235.0     | 1636.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1715.0     | 1270.370 | 1.270  | 1306.0     | 1733.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.9      | 426.568  | 0.427  | 459.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.778  | 0.274  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1734.3     | 1284.667 | 1.285  | 1694.0     | 1730.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.9       | 14.716   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1656.5     | 1227.062 | 1.227  | 1459.0     | 1619.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1749.0     | 1295.531 | 1.296  | 1541.0     | 1714.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.3      | 422.469  | 0.422  | 425.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 370.4      | 274.346  | 0.274  | 364.0      | 371.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1687.6     | 1250.086 | 1.250  | 1640.9                | 1687.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.9       | 16.235   | 0.016  | 19.9                  | 21.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1639.5     | 1214.481 | 1.214  | 1622.6                | 1639.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1732.0     | 1282.951 | 1.283  | 1715.0                | 1732.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.1      | 424.519  | 0.425  | 570.3                 | 573.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 370.0      | 274.062  | 0.274  | 369.6                 | 370.0                    |

Final op time: 1.283 us (1732.0 cycles), bucket=TRISC_0 compute

