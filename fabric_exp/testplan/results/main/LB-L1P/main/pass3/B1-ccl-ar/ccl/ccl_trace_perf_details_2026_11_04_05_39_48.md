# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_39_48
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_04_05_39_59/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1647.4     | 1220.296 | 1.220  | 1600.0     | 1625.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.5       | 18.148   | 0.018  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1651.0     | 1222.938 | 1.223  | 1502.0     | 1643.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1745.0     | 1292.593 | 1.293  | 1587.0     | 1739.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.5      | 421.086  | 0.421  | 445.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.802  | 0.274  | 365.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1738.6     | 1287.852 | 1.288  | 1663.0     | 1729.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.7       | 13.852   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1631.0     | 1208.148 | 1.208  | 1442.0     | 1618.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1722.3     | 1275.802 | 1.276  | 1539.0     | 1704.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.1      | 423.802  | 0.424  | 433.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.753  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1693.0     | 1254.074 | 1.254  | 1647.4                | 1693.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.6       | 16.000   | 0.016  | 18.7                  | 21.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1641.0     | 1215.543 | 1.216  | 1631.0                | 1641.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1733.7     | 1284.198 | 1.284  | 1722.3                | 1733.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.3      | 422.444  | 0.422  | 568.5                 | 570.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.778  | 0.274  | 369.6                 | 369.6                    |

Final op time: 1.284 us (1733.7 cycles), bucket=TRISC_0 compute

