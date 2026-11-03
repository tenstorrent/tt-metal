# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_13_59
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

Report: generated/profiler/reports/2026_11_02_09_14_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1866.8     | 1382.790 | 1.383  | 1831.0     | 1853.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.3       | 21.728   | 0.022  | 18.0       | 27.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1812.7     | 1342.716 | 1.343  | 1638.0     | 1817.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1940.8     | 1437.605 | 1.438  | 1763.0     | 1935.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.8      | 422.815  | 0.423  | 415.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.0      | 277.062  | 0.277  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1946.4     | 1441.802 | 1.442  | 1886.0     | 1913.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.358   | 0.015  | 18.0       | 18.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1803.5     | 1335.901 | 1.336  | 1608.0     | 1767.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1941.1     | 1437.852 | 1.438  | 1750.0     | 1900.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 559.8      | 414.667  | 0.415  | 401.0      | 562.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.2      | 277.951  | 0.278  | 372.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1906.6     | 1412.296 | 1.412  | 1866.8                | 1906.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.0       | 18.543   | 0.019  | 20.7                  | 25.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1808.1     | 1339.309 | 1.339  | 1803.5                | 1808.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1940.9     | 1437.728 | 1.438  | 1940.8                | 1940.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 565.3      | 418.741  | 0.419  | 559.8                 | 565.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.6      | 277.506  | 0.278  | 374.0                 | 374.6                    |

Final op time: 1.438 us (1940.9 cycles), bucket=TRISC_0 compute

