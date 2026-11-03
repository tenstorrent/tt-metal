# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_36_06
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_36_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1627.0     | 1205.160 | 1.205  | 1580.0     | 1610.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.3       | 20.247   | 0.020  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1594.8     | 1181.358 | 1.181  | 1441.0     | 1592.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1690.6     | 1252.296 | 1.252  | 1539.0     | 1692.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 576.5      | 427.012  | 0.427  | 413.0      | 586.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.1      | 277.852  | 0.278  | 366.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1696.0     | 1256.272 | 1.256  | 1653.0     | 1690.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.8       | 15.407   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1584.9     | 1174.000 | 1.174  | 1452.0     | 1548.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1679.0     | 1243.728 | 1.244  | 1540.0     | 1651.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.6      | 423.432  | 0.423  | 432.0      | 583.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.802  | 0.278  | 368.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1661.5     | 1230.716 | 1.231  | 1627.0                | 1661.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.1       | 17.827   | 0.018  | 20.8                  | 24.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1589.9     | 1177.679 | 1.178  | 1584.9                | 1589.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1684.8     | 1248.012 | 1.248  | 1679.0                | 1684.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.0      | 425.222  | 0.425  | 571.6                 | 574.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.1      | 277.827  | 0.278  | 375.0                 | 375.1                    |

Final op time: 1.248 us (1684.8 cycles), bucket=TRISC_0 compute

