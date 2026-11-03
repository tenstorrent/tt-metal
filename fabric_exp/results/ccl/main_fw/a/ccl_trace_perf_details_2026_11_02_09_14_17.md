# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_14_17
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

Report: generated/profiler/reports/2026_11_02_09_14_33/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1621.8     | 1201.309 | 1.201  | 1575.0     | 1609.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 23.6       | 17.457   | 0.017  | 18.0       | 20.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1553.7     | 1150.889 | 1.151  | 1448.0     | 1545.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1684.4     | 1247.728 | 1.248  | 1576.0     | 1680.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 566.1      | 419.333  | 0.419  | 468.0      | 567.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.778  | 0.278  | 370.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1683.6     | 1247.086 | 1.247  | 1653.0     | 1676.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.333   | 0.015  | 18.0       | 19.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1544.7     | 1144.198 | 1.144  | 1389.0     | 1536.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1681.3     | 1245.383 | 1.245  | 1508.0     | 1675.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 557.5      | 412.988  | 0.413  | 407.0      | 563.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.7      | 277.580  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1652.7     | 1224.198 | 1.224  | 1621.8                | 1652.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.1       | 16.395   | 0.016  | 20.7                  | 22.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1549.2     | 1147.543 | 1.148  | 1544.7                | 1549.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1682.8     | 1246.556 | 1.247  | 1681.3                | 1682.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 561.8      | 416.160  | 0.416  | 557.5                 | 561.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.679  | 0.278  | 374.7                 | 374.9                    |

Final op time: 1.247 us (1682.8 cycles), bucket=TRISC_0 compute

