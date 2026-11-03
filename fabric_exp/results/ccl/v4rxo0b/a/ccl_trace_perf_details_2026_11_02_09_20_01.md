# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_20_01
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

Report: generated/profiler/reports/2026_11_02_09_20_21/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1858.2     | 1376.469 | 1.376  | 1829.0     | 1840.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.8       | 19.111   | 0.019  | 18.0       | 19.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1801.9     | 1334.741 | 1.335  | 1334.0     | 1806.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1933.8     | 1432.444 | 1.432  | 1479.0     | 1947.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.4      | 420.296  | 0.420  | 418.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.753  | 0.278  | 368.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1937.6     | 1435.235 | 1.435  | 1910.0     | 1932.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.4       | 15.086   | 0.015  | 18.0       | 18.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1794.7     | 1329.432 | 1.329  | 1664.0     | 1771.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1925.0     | 1425.901 | 1.426  | 1790.0     | 1907.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 564.8      | 418.370  | 0.418  | 438.0      | 567.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.3      | 277.284  | 0.277  | 365.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1897.9     | 1405.852 | 1.406  | 1858.2                | 1897.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.1       | 17.099   | 0.017  | 20.4                  | 23.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1798.3     | 1332.086 | 1.332  | 1794.7                | 1798.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1929.4     | 1429.173 | 1.429  | 1925.0                | 1929.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 566.1      | 419.333  | 0.419  | 564.8                 | 566.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.6      | 277.519  | 0.278  | 374.3                 | 374.6                    |

Final op time: 1.429 us (1929.4 cycles), bucket=TRISC_0 compute

