# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_14_49
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

Report: generated/profiler/reports/2026_11_02_09_15_04/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1641.1     | 1215.605 | 1.216  | 1580.0     | 1635.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.1       | 19.309   | 0.019  | 18.0       | 23.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1597.5     | 1183.309 | 1.183  | 1428.0     | 1598.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1729.0     | 1280.741 | 1.281  | 1552.0     | 1727.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.8      | 420.593  | 0.421  | 417.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.605  | 0.278  | 370.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1728.6     | 1280.469 | 1.280  | 1662.0     | 1727.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.358   | 0.015  | 18.0       | 19.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1587.1     | 1175.654 | 1.176  | 1431.0     | 1569.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1724.1     | 1277.086 | 1.277  | 1566.0     | 1704.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 559.0      | 414.074  | 0.414  | 402.0      | 562.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.1      | 277.852  | 0.278  | 369.0      | 376.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1684.8     | 1248.037 | 1.248  | 1641.1                | 1684.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.4       | 17.333   | 0.017  | 20.7                  | 23.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1592.3     | 1179.481 | 1.179  | 1587.1                | 1592.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1726.5     | 1278.914 | 1.279  | 1724.1                | 1726.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 563.4      | 417.333  | 0.417  | 559.0                 | 563.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.728  | 0.278  | 374.8                 | 374.9                    |

Final op time: 1.279 us (1726.5 cycles), bucket=TRISC_0 compute

