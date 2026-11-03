# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_35_00
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

Report: generated/profiler/reports/2026_11_02_09_35_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1635.1     | 1211.160 | 1.211  | 1579.0     | 1621.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.0       | 20.741   | 0.021  | 18.0       | 24.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1568.6     | 1161.926 | 1.162  | 1490.0     | 1559.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1700.4     | 1259.531 | 1.260  | 1616.0     | 1694.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.9      | 420.691  | 0.421  | 411.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.7      | 277.580  | 0.278  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1695.9     | 1256.247 | 1.256  | 1653.0     | 1695.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.1       | 14.889   | 0.015  | 18.0       | 19.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1556.7     | 1153.136 | 1.153  | 1409.0     | 1539.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1694.8     | 1255.383 | 1.255  | 1538.0     | 1669.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 557.3      | 412.790  | 0.413  | 426.0      | 562.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.753  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1665.5     | 1233.704 | 1.234  | 1635.1                | 1665.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.1       | 17.815   | 0.018  | 20.1                  | 24.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1562.7     | 1157.531 | 1.158  | 1556.7                | 1562.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1697.6     | 1257.457 | 1.257  | 1694.8                | 1697.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 562.6      | 416.741  | 0.417  | 557.3                 | 562.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.667  | 0.278  | 374.7                 | 374.9                    |

Final op time: 1.257 us (1697.6 cycles), bucket=TRISC_0 compute

