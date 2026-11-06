# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_29_00
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_04_09_29_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1755.7     | 1300.543 | 1.301  | 1694.0     | 1731.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.1       | 14.123   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1600.6     | 1185.605 | 1.186  | 1197.0     | 1602.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1734.2     | 1284.617 | 1.285  | 1342.0     | 1740.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.7      | 424.222  | 0.424  | 389.0      | 577.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.580  | 0.274  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1685.5     | 1248.494 | 1.248  | 1643.0     | 1686.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 35.3       | 26.173   | 0.026  | 18.0       | 34.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1685.3     | 1248.346 | 1.248  | 1497.0     | 1612.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1816.2     | 1345.333 | 1.345  | 1630.0     | 1746.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.1      | 423.012  | 0.423  | 403.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.679  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1720.6     | 1274.519 | 1.275  | 1685.5                | 1720.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 27.2       | 20.148   | 0.020  | 19.1                  | 27.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1642.9     | 1216.975 | 1.217  | 1600.6                | 1642.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1775.2     | 1314.975 | 1.315  | 1734.2                | 1775.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.9      | 423.617  | 0.424  | 571.1                 | 571.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.630  | 0.274  | 369.3                 | 369.4                    |

Final op time: 1.315 us (1775.2 cycles), bucket=TRISC_0 compute

