# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_53_15
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

Report: generated/profiler/reports/2026_11_02_09_53_28/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1860.7     | 1378.272 | 1.378  | 1824.0     | 1847.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.5       | 19.630   | 0.020  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1796.3     | 1330.617 | 1.331  | 1703.0     | 1790.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1930.4     | 1429.926 | 1.430  | 1859.0     | 1929.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.7      | 420.519  | 0.421  | 428.0      | 572.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 373.8      | 276.864  | 0.277  | 368.0      | 373.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1927.3     | 1427.605 | 1.428  | 1906.0     | 1923.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.1       | 14.889   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1786.3     | 1323.210 | 1.323  | 1641.0     | 1772.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1920.9     | 1422.914 | 1.423  | 1766.0     | 1909.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 558.2      | 413.506  | 0.414  | 385.0      | 565.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.7      | 277.531  | 0.278  | 369.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1894.0     | 1402.938 | 1.403  | 1860.7                | 1894.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.3       | 17.259   | 0.017  | 20.1                  | 23.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1791.3     | 1326.914 | 1.327  | 1786.3                | 1791.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1925.7     | 1426.420 | 1.426  | 1920.9                | 1925.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 563.0      | 417.012  | 0.417  | 558.2                 | 563.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.2      | 277.198  | 0.277  | 373.8                 | 374.2                    |

Final op time: 1.426 us (1925.7 cycles), bucket=TRISC_0 compute

