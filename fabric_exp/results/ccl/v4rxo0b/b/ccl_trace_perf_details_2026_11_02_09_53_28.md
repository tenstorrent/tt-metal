# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_53_28
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

Report: generated/profiler/reports/2026_11_02_09_53_41/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1630.0     | 1207.432 | 1.207  | 1574.0     | 1612.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.7       | 19.037   | 0.019  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1564.8     | 1159.086 | 1.159  | 1496.0     | 1560.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1695.4     | 1255.852 | 1.256  | 1628.0     | 1692.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 566.8      | 419.877  | 0.420  | 429.0      | 570.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.2      | 277.210  | 0.277  | 367.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1695.5     | 1255.926 | 1.256  | 1661.0     | 1690.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.8       | 15.407   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1550.2     | 1148.296 | 1.148  | 1411.0     | 1542.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1685.8     | 1248.741 | 1.249  | 1530.0     | 1680.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.6      | 415.259  | 0.415  | 399.0      | 563.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.3      | 277.975  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1662.8     | 1231.679 | 1.232  | 1630.0                | 1662.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.2       | 17.222   | 0.017  | 20.8                  | 23.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1557.5     | 1153.691 | 1.154  | 1550.2                | 1557.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1690.6     | 1252.296 | 1.252  | 1685.8                | 1690.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 563.7      | 417.568  | 0.418  | 560.6                 | 563.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.593  | 0.278  | 374.2                 | 374.8                    |

Final op time: 1.252 us (1690.6 cycles), bucket=TRISC_0 compute

