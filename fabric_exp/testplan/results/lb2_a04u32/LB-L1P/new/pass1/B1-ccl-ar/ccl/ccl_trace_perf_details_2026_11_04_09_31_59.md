# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_31_59
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

Report: generated/profiler/reports/2026_11_04_09_32_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1899.6     | 1407.136 | 1.407  | 1848.0     | 1910.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.9       | 19.901   | 0.020  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1854.9     | 1373.975 | 1.374  | 1677.0     | 1858.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1983.3     | 1469.136 | 1.469  | 1808.0     | 1985.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 565.4      | 418.790  | 0.419  | 403.0      | 568.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 273.975  | 0.274  | 363.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1982.5     | 1468.519 | 1.469  | 1936.0     | 1977.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.2       | 14.963   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1836.1     | 1360.049 | 1.360  | 1714.0     | 1817.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1968.3     | 1457.975 | 1.458  | 1832.0     | 1950.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 566.5      | 419.630  | 0.420  | 432.0      | 566.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1941.1     | 1437.827 | 1.438  | 1899.6                | 1941.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.432   | 0.017  | 20.2                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1845.5     | 1367.012 | 1.367  | 1836.1                | 1845.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1975.8     | 1463.556 | 1.464  | 1968.3                | 1975.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 565.9      | 419.210  | 0.419  | 565.4                 | 565.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.704  | 0.274  | 369.1                 | 369.5                    |

Final op time: 1.464 us (1975.8 cycles), bucket=TRISC_0 compute

