# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_47_18
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_47_31/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1627.3     | 1205.383 | 1.205  | 1576.0     | 1613.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.4       | 21.037   | 0.021  | 18.0       | 27.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1580.8     | 1170.938 | 1.171  | 1506.0     | 1568.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1713.9     | 1269.556 | 1.270  | 1634.0     | 1698.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 562.9      | 416.963  | 0.417  | 407.0      | 568.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.9      | 278.444  | 0.278  | 370.0      | 376.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1704.7     | 1262.716 | 1.263  | 1652.0     | 1695.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.4       | 15.111   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1563.5     | 1158.148 | 1.158  | 1411.0     | 1533.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1697.6     | 1257.457 | 1.257  | 1551.0     | 1670.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.7      | 415.333  | 0.415  | 451.0      | 563.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.457  | 0.277  | 370.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1666.0     | 1234.049 | 1.234  | 1627.3                | 1666.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.4       | 18.074   | 0.018  | 20.4                  | 24.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1572.1     | 1164.543 | 1.165  | 1563.5                | 1572.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1705.7     | 1263.506 | 1.264  | 1697.6                | 1705.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 561.8      | 416.148  | 0.416  | 560.7                 | 561.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.2      | 277.951  | 0.278  | 374.6                 | 375.2                    |

Final op time: 1.264 us (1705.7 cycles), bucket=TRISC_0 compute

