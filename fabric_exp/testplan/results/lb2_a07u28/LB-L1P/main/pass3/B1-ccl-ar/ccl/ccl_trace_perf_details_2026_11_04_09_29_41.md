# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_29_41
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_09_29_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1787.9     | 1324.370 | 1.324  | 1684.0     | 1799.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.6       | 13.778   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1666.1     | 1234.148 | 1.234  | 1211.0     | 1687.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1759.2     | 1303.086 | 1.303  | 1313.0     | 1780.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.6      | 421.160  | 0.421  | 432.0      | 571.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.383  | 0.273  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1710.7     | 1267.185 | 1.267  | 1619.0     | 1702.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 36.2       | 26.815   | 0.027  | 22.0       | 36.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1749.7     | 1296.049 | 1.296  | 1559.0     | 1708.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1842.9     | 1365.111 | 1.365  | 1642.0     | 1801.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.4      | 423.975  | 0.424  | 444.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.654  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1749.3     | 1295.778 | 1.296  | 1710.7                | 1749.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 27.4       | 20.296   | 0.020  | 18.6                  | 27.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1707.9     | 1265.099 | 1.265  | 1666.1                | 1707.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1801.0     | 1334.099 | 1.334  | 1759.2                | 1801.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.5      | 422.568  | 0.423  | 568.6                 | 570.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.519  | 0.274  | 369.1                 | 369.2                    |

Final op time: 1.334 us (1801.0 cycles), bucket=TRISC_0 compute

