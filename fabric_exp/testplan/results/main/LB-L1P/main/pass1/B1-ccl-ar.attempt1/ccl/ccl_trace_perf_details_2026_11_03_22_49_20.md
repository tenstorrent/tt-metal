# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_49_20
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

Report: generated/profiler/reports/2026_11_03_22_49_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1896.5     | 1404.790 | 1.405  | 1853.0     | 1898.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.5       | 21.136   | 0.021  | 18.0       | 25.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1879.3     | 1392.099 | 1.392  | 1575.0     | 1898.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 2012.8     | 1490.938 | 1.491  | 1709.0     | 2032.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.7      | 420.494  | 0.420  | 443.0      | 572.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 2012.6     | 1490.790 | 1.491  | 1922.0     | 2013.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.0       | 14.074   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1857.9     | 1376.247 | 1.376  | 1675.0     | 1854.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1985.8     | 1470.963 | 1.471  | 1813.0     | 1969.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.3      | 425.383  | 0.425  | 470.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.605  | 0.274  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1954.5     | 1447.790 | 1.448  | 1896.5                | 1954.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.8       | 17.605   | 0.018  | 19.0                  | 23.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1868.6     | 1384.173 | 1.384  | 1857.9                | 1868.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1999.3     | 1480.951 | 1.481  | 1985.8                | 1999.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.0      | 422.938  | 0.423  | 567.7                 | 571.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.568  | 0.274  | 369.3                 | 369.3                    |

Final op time: 1.481 us (1999.3 cycles), bucket=TRISC_0 compute

