# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_59_56
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

Report: generated/profiler/reports/2026_11_02_10_00_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1863.1     | 1380.049 | 1.380  | 1824.0     | 1850.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.2       | 19.383   | 0.019  | 18.0       | 21.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1808.4     | 1339.580 | 1.340  | 1342.0     | 1853.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1934.1     | 1432.642 | 1.433  | 1467.0     | 1979.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.1      | 425.235  | 0.425  | 433.0      | 577.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 373.8      | 276.864  | 0.277  | 368.0      | 373.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1950.6     | 1444.889 | 1.445  | 1886.0     | 1926.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.8       | 15.432   | 0.015  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1813.7     | 1343.481 | 1.343  | 1576.0     | 1765.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1946.3     | 1441.728 | 1.442  | 1720.0     | 1896.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.8      | 415.432  | 0.415  | 396.0      | 565.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.605  | 0.278  | 369.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1906.8     | 1412.469 | 1.412  | 1863.1                | 1906.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.407   | 0.017  | 20.8                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1811.1     | 1341.531 | 1.342  | 1808.4                | 1811.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1940.2     | 1437.185 | 1.437  | 1934.1                | 1940.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 567.5      | 420.333  | 0.420  | 560.8                 | 567.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.3      | 277.235  | 0.277  | 373.8                 | 374.3                    |

Final op time: 1.437 us (1940.2 cycles), bucket=TRISC_0 compute

