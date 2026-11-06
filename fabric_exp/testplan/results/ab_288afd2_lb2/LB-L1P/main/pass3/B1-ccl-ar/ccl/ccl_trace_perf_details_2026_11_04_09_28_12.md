# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_12
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

Report: generated/profiler/reports/2026_11_04_09_28_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 2020.1     | 1496.370 | 1.496  | 1992.0     | 2017.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.901   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1853.8     | 1373.160 | 1.373  | 1510.0     | 1866.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1981.3     | 1467.654 | 1.468  | 1643.0     | 1993.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.8      | 425.012  | 0.425  | 427.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.556  | 0.274  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1918.8     | 1421.358 | 1.421  | 1866.0     | 1923.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 34.9       | 25.852   | 0.026  | 19.0       | 33.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1931.1     | 1430.469 | 1.430  | 1806.0     | 1897.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 2056.6     | 1523.407 | 1.523  | 1927.0     | 2017.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.9      | 422.123  | 0.422  | 438.0      | 576.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.827  | 0.274  | 363.0      | 370.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1969.5     | 1458.864 | 1.459  | 1918.8                | 1969.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 26.8       | 19.877   | 0.020  | 18.8                  | 26.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1892.5     | 1401.815 | 1.402  | 1853.8                | 1892.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 2019.0     | 1495.531 | 1.496  | 1981.3                | 2019.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.8      | 423.568  | 0.424  | 569.9                 | 571.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.691  | 0.274  | 369.3                 | 369.5                    |

Final op time: 1.496 us (2019.0 cycles), bucket=TRISC_0 compute

