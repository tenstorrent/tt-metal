# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_27_02
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

Report: generated/profiler/reports/2026_11_04_05_27_20/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1901.9     | 1408.815 | 1.409  | 1850.0     | 1911.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.1       | 21.556   | 0.022  | 18.0       | 28.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1879.0     | 1391.827 | 1.392  | 1747.0     | 1891.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 2015.1     | 1492.667 | 1.493  | 1892.0     | 2026.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.0      | 420.716  | 0.421  | 451.0      | 570.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.6      | 274.543  | 0.275  | 363.0      | 371.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 2009.5     | 1488.519 | 1.489  | 1909.0     | 2017.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.4       | 13.630   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1855.5     | 1374.444 | 1.374  | 1669.0     | 1847.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1981.2     | 1467.556 | 1.468  | 1794.0     | 1964.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.4      | 424.000  | 0.424  | 424.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.630  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1955.7     | 1448.667 | 1.449  | 1901.9                | 1955.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.8       | 17.593   | 0.018  | 18.4                  | 23.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1867.2     | 1383.136 | 1.383  | 1855.5                | 1867.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1998.2     | 1480.111 | 1.480  | 1981.2                | 1998.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.2      | 422.358  | 0.422  | 568.0                 | 570.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 370.0      | 274.086  | 0.274  | 369.4                 | 370.0                    |

Final op time: 1.480 us (1998.2 cycles), bucket=TRISC_0 compute

