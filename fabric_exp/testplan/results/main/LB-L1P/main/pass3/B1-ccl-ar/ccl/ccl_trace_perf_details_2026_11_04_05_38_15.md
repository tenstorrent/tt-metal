# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_38_15
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

Report: generated/profiler/reports/2026_11_04_05_38_25/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1897.2     | 1405.309 | 1.405  | 1850.0     | 1905.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 30.0       | 22.198   | 0.022  | 18.0       | 29.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1882.0     | 1394.099 | 1.394  | 1608.0     | 1898.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 2009.9     | 1488.815 | 1.489  | 1724.0     | 2022.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.9      | 422.889  | 0.423  | 447.0      | 576.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.556  | 0.274  | 363.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 2010.4     | 1489.210 | 1.489  | 1964.0     | 2014.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.0       | 14.099   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1858.3     | 1376.519 | 1.377  | 1642.0     | 1837.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1991.7     | 1475.358 | 1.475  | 1774.0     | 1981.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.5      | 424.099  | 0.424  | 457.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.630  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1953.8     | 1447.259 | 1.447  | 1897.2                | 1953.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.5       | 18.148   | 0.018  | 19.0                  | 24.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1870.2     | 1385.309 | 1.385  | 1858.3                | 1870.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 2000.8     | 1482.086 | 1.482  | 1991.7                | 2000.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.7      | 423.494  | 0.423  | 570.9                 | 571.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.593  | 0.274  | 369.3                 | 369.4                    |

Final op time: 1.482 us (2000.8 cycles), bucket=TRISC_0 compute

