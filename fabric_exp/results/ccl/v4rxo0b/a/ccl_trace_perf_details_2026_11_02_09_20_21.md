# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_20_21
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_20_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1620.7     | 1200.543 | 1.201  | 1580.0     | 1609.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.3       | 18.025   | 0.018  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1557.8     | 1153.926 | 1.154  | 1128.0     | 1567.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1687.5     | 1249.975 | 1.250  | 1253.0     | 1700.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.0      | 421.506  | 0.422  | 462.0      | 570.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.2      | 277.951  | 0.278  | 370.0      | 375.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1700.7     | 1259.778 | 1.260  | 1662.0     | 1697.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.8       | 16.148   | 0.016  | 18.0       | 20.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1561.3     | 1156.519 | 1.157  | 1434.0     | 1521.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1693.7     | 1254.593 | 1.255  | 1565.0     | 1649.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 563.3      | 417.235  | 0.417  | 436.0      | 563.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.457  | 0.277  | 365.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1660.7     | 1230.160 | 1.230  | 1620.7                | 1660.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.1       | 17.086   | 0.017  | 21.8                  | 23.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1559.5     | 1155.222 | 1.155  | 1557.8                | 1559.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1690.6     | 1252.284 | 1.252  | 1687.5                | 1690.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 566.1      | 419.370  | 0.419  | 563.3                 | 566.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.704  | 0.278  | 374.6                 | 374.9                    |

Final op time: 1.252 us (1690.6 cycles), bucket=TRISC_0 compute

