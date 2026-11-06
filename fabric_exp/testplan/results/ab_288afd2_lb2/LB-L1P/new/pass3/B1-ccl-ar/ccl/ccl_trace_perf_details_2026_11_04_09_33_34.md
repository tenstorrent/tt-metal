# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_33_34
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_09_33_53/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1739.6     | 1288.568 | 1.289  | 1707.0     | 1732.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.1       | 14.148   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1588.7     | 1176.840 | 1.177  | 1316.0     | 1593.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1719.8     | 1273.926 | 1.274  | 1437.0     | 1731.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.2      | 423.852  | 0.424  | 428.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.580  | 0.274  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1674.3     | 1240.198 | 1.240  | 1619.0     | 1671.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 34.0       | 25.210   | 0.025  | 18.0       | 35.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1647.6     | 1220.420 | 1.220  | 1508.0     | 1613.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1782.3     | 1320.222 | 1.320  | 1657.0     | 1748.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.9      | 424.395  | 0.424  | 435.0      | 576.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.605  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1706.9     | 1264.383 | 1.264  | 1674.3                | 1706.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 26.6       | 19.679   | 0.020  | 19.1                  | 26.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1618.2     | 1198.630 | 1.199  | 1588.7                | 1618.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1751.0     | 1297.074 | 1.297  | 1719.8                | 1751.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.6      | 424.123  | 0.424  | 572.2                 | 572.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.593  | 0.274  | 369.3                 | 369.4                    |

Final op time: 1.297 us (1751.0 cycles), bucket=TRISC_0 compute

