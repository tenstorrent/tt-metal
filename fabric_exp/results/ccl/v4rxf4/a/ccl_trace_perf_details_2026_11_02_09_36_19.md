# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_36_19
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

Report: generated/profiler/reports/2026_11_02_09_36_32/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1634.9     | 1211.062 | 1.211  | 1574.0     | 1637.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.4       | 21.037   | 0.021  | 18.0       | 24.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1607.5     | 1190.765 | 1.191  | 1541.0     | 1599.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1702.3     | 1260.963 | 1.261  | 1627.0     | 1701.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.6      | 424.173  | 0.424  | 412.0      | 581.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.8      | 278.346  | 0.278  | 365.0      | 376.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1705.2     | 1263.111 | 1.263  | 1666.0     | 1698.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.2       | 14.963   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1591.0     | 1178.519 | 1.179  | 1461.0     | 1571.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1684.7     | 1247.951 | 1.248  | 1553.0     | 1672.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.6      | 424.148  | 0.424  | 436.0      | 584.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.4      | 277.358  | 0.277  | 368.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1670.1     | 1237.086 | 1.237  | 1634.9                | 1670.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.3       | 18.000   | 0.018  | 20.2                  | 24.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1599.3     | 1184.642 | 1.185  | 1591.0                | 1599.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1693.5     | 1254.457 | 1.254  | 1684.7                | 1693.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.6      | 424.160  | 0.424  | 572.6                 | 572.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.1      | 277.852  | 0.278  | 374.4                 | 375.1                    |

Final op time: 1.254 us (1693.5 cycles), bucket=TRISC_0 compute

