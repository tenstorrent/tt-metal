# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_35_18
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

Report: generated/profiler/reports/2026_11_04_09_35_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1749.1     | 1295.605 | 1.296  | 1706.0     | 1736.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.926   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1633.6     | 1210.049 | 1.210  | 1123.0     | 1645.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1729.3     | 1280.988 | 1.281  | 1227.0     | 1735.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 564.6      | 418.222  | 0.418  | 425.0      | 572.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.8      | 273.160  | 0.273  | 366.0      | 368.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1682.0     | 1245.926 | 1.246  | 1600.0     | 1686.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 34.9       | 25.877   | 0.026  | 18.0       | 33.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1685.5     | 1248.494 | 1.248  | 1591.0     | 1655.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1775.5     | 1315.160 | 1.315  | 1673.0     | 1739.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.5      | 425.531  | 0.426  | 471.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.333  | 0.273  | 365.0      | 368.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1715.5     | 1270.765 | 1.271  | 1682.0                | 1715.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 26.9       | 19.901   | 0.020  | 18.8                  | 26.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1659.5     | 1229.272 | 1.229  | 1633.6                | 1659.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1752.4     | 1298.074 | 1.298  | 1729.3                | 1752.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.5      | 421.877  | 0.422  | 564.6                 | 569.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.9      | 273.247  | 0.273  | 368.8                 | 368.9                    |

Final op time: 1.298 us (1752.4 cycles), bucket=TRISC_0 compute

