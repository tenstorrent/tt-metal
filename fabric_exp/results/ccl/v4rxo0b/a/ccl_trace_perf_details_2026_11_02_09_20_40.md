# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_20_40
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_20_58/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1629.5     | 1207.037 | 1.207  | 1577.0     | 1617.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.7       | 19.802   | 0.020  | 18.0       | 23.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1569.0     | 1162.247 | 1.162  | 1500.0     | 1564.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1700.1     | 1259.333 | 1.259  | 1633.0     | 1694.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 564.2      | 417.951  | 0.418  | 374.0      | 568.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.679  | 0.278  | 370.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1693.0     | 1254.049 | 1.254  | 1653.0     | 1693.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.309   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1553.1     | 1150.469 | 1.150  | 1397.0     | 1546.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1686.1     | 1248.988 | 1.249  | 1522.0     | 1684.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 558.1      | 413.432  | 0.413  | 386.0      | 562.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.704  | 0.278  | 372.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1661.2     | 1230.543 | 1.231  | 1629.5                | 1661.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.7       | 17.556   | 0.018  | 20.7                  | 23.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1561.1     | 1156.358 | 1.156  | 1553.1                | 1561.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1693.1     | 1254.160 | 1.254  | 1686.1                | 1693.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 561.2      | 415.691  | 0.416  | 558.1                 | 561.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.691  | 0.278  | 374.9                 | 374.9                    |

Final op time: 1.254 us (1693.1 cycles), bucket=TRISC_0 compute

