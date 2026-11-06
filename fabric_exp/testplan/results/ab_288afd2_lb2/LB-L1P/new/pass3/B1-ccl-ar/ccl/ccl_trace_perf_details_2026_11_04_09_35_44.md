# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_35_44
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

Report: generated/profiler/reports/2026_11_04_09_35_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1739.8     | 1288.741 | 1.289  | 1699.0     | 1732.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.0       | 14.099   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1629.2     | 1206.815 | 1.207  | 1349.0     | 1635.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1724.4     | 1277.333 | 1.277  | 1453.0     | 1731.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 566.1      | 419.309  | 0.419  | 438.0      | 572.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.309  | 0.273  | 365.0      | 368.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1682.8     | 1246.519 | 1.247  | 1600.0     | 1686.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 35.6       | 26.395   | 0.026  | 18.0       | 35.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1688.4     | 1250.691 | 1.251  | 1588.0     | 1653.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1780.6     | 1318.963 | 1.319  | 1671.0     | 1745.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.1      | 421.580  | 0.422  | 442.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.802  | 0.274  | 364.0      | 369.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1711.3     | 1267.630 | 1.268  | 1682.8                | 1711.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 27.3       | 20.247   | 0.020  | 19.0                  | 27.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1658.8     | 1228.753 | 1.229  | 1629.2                | 1658.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1752.5     | 1298.148 | 1.298  | 1724.4                | 1752.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 567.6      | 420.444  | 0.420  | 566.1                 | 567.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.556  | 0.274  | 369.0                 | 369.3                    |

Final op time: 1.298 us (1752.5 cycles), bucket=TRISC_0 compute

