# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_46_27
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

Report: generated/profiler/reports/2026_11_02_09_46_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1857.1     | 1375.654 | 1.376  | 1827.0     | 1843.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.4       | 19.531   | 0.020  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1802.5     | 1335.185 | 1.335  | 1632.0     | 1802.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1935.5     | 1433.704 | 1.434  | 1748.0     | 1948.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 563.3      | 417.284  | 0.417  | 401.0      | 568.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.2      | 277.210  | 0.277  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1932.3     | 1431.309 | 1.431  | 1901.0     | 1927.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.6       | 15.235   | 0.015  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1787.0     | 1323.679 | 1.324  | 1680.0     | 1775.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1921.6     | 1423.432 | 1.423  | 1795.0     | 1913.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 563.8      | 417.654  | 0.418  | 440.0      | 569.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.2      | 277.210  | 0.277  | 371.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1894.7     | 1403.481 | 1.403  | 1857.1                | 1894.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.383   | 0.017  | 20.6                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1794.7     | 1329.432 | 1.329  | 1787.0                | 1794.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1928.6     | 1428.568 | 1.429  | 1921.6                | 1928.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 563.6      | 417.469  | 0.417  | 563.3                 | 563.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.2      | 277.210  | 0.277  | 374.2                 | 374.2                    |

Final op time: 1.429 us (1928.6 cycles), bucket=TRISC_0 compute

