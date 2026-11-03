# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_35_36
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

Report: generated/profiler/reports/2026_11_02_09_35_52/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1631.9     | 1208.790 | 1.209  | 1577.0     | 1630.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.6       | 18.988   | 0.019  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1572.9     | 1165.136 | 1.165  | 1464.0     | 1575.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1702.9     | 1261.432 | 1.261  | 1598.0     | 1705.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.5      | 421.111  | 0.421  | 436.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.654  | 0.278  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1699.2     | 1258.667 | 1.259  | 1662.0     | 1696.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.5       | 15.160   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1559.4     | 1155.136 | 1.155  | 1426.0     | 1539.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1695.2     | 1255.704 | 1.256  | 1563.0     | 1677.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.7      | 415.333  | 0.415  | 479.0      | 564.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.778  | 0.278  | 372.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1665.5     | 1233.728 | 1.234  | 1631.9                | 1665.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.0       | 17.074   | 0.017  | 20.5                  | 23.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1566.2     | 1160.136 | 1.160  | 1559.4                | 1566.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1699.1     | 1258.568 | 1.259  | 1695.2                | 1699.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 564.6      | 418.222  | 0.418  | 560.7                 | 564.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.716  | 0.278  | 374.8                 | 374.9                    |

Final op time: 1.259 us (1699.1 cycles), bucket=TRISC_0 compute

