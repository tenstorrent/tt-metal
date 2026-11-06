# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_59_12
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

Report: generated/profiler/reports/2026_11_03_22_59_25/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1630.7     | 1207.926 | 1.208  | 1593.0     | 1619.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.0       | 18.494   | 0.018  | 18.0       | 19.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1622.4     | 1201.753 | 1.202  | 1167.0     | 1637.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1716.3     | 1271.358 | 1.271  | 1267.0     | 1734.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.6      | 421.185  | 0.421  | 468.0      | 571.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.4      | 274.395  | 0.274  | 366.0      | 370.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1725.4     | 1278.099 | 1.278  | 1685.0     | 1723.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.7       | 14.617   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1632.6     | 1209.333 | 1.209  | 1418.0     | 1561.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1725.6     | 1278.222 | 1.278  | 1500.0     | 1667.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.0      | 422.198  | 0.422  | 457.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.8      | 273.926  | 0.274  | 364.0      | 370.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1678.1     | 1243.012 | 1.243  | 1630.7                | 1678.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.4       | 16.556   | 0.017  | 19.7                  | 22.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1627.5     | 1205.543 | 1.206  | 1622.4                | 1627.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1721.0     | 1274.790 | 1.275  | 1716.3                | 1721.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.3      | 421.691  | 0.422  | 568.6                 | 569.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 370.1      | 274.160  | 0.274  | 369.8                 | 370.1                    |

Final op time: 1.275 us (1721.0 cycles), bucket=TRISC_0 compute

