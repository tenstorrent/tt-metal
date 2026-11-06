# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_34_05
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_05_34_15/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1647.4     | 1220.321 | 1.220  | 1600.0     | 1633.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.9       | 21.432   | 0.021  | 18.0       | 27.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1641.8     | 1216.173 | 1.216  | 1463.0     | 1642.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1737.2     | 1286.840 | 1.287  | 1561.0     | 1737.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.8      | 421.358  | 0.421  | 436.0      | 572.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.3      | 274.296  | 0.274  | 366.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1730.5     | 1281.852 | 1.282  | 1697.0     | 1730.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.9       | 14.741   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1624.0     | 1202.988 | 1.203  | 1425.0     | 1629.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1720.2     | 1274.247 | 1.274  | 1518.0     | 1728.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.1      | 420.049  | 0.420  | 435.0      | 570.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.728  | 0.274  | 364.0      | 368.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1689.0     | 1251.086 | 1.251  | 1647.4                | 1689.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.4       | 18.086   | 0.018  | 19.9                  | 24.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1632.9     | 1209.580 | 1.210  | 1624.0                | 1632.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1728.7     | 1280.543 | 1.281  | 1720.2                | 1728.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 568.0      | 420.704  | 0.421  | 567.1                 | 568.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.9      | 274.012  | 0.274  | 369.5                 | 369.9                    |

Final op time: 1.281 us (1728.7 cycles), bucket=TRISC_0 compute

