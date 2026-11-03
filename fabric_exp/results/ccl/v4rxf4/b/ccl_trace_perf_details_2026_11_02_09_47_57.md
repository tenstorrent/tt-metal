# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_47_57
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

Report: generated/profiler/reports/2026_11_02_09_48_10/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1621.9     | 1201.407 | 1.201  | 1580.0     | 1615.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.7       | 18.272   | 0.018  | 18.0       | 21.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1599.2     | 1184.568 | 1.185  | 1227.0     | 1600.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1691.4     | 1252.914 | 1.253  | 1315.0     | 1697.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 579.8      | 429.457  | 0.429  | 440.0      | 586.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.728  | 0.278  | 372.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1704.4     | 1262.519 | 1.263  | 1652.0     | 1701.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.309   | 0.015  | 18.0       | 19.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1597.9     | 1183.630 | 1.184  | 1434.0     | 1552.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1692.7     | 1253.877 | 1.254  | 1527.0     | 1649.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 564.9      | 418.420  | 0.418  | 424.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.457  | 0.277  | 368.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1663.2     | 1231.963 | 1.232  | 1621.9                | 1663.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.7       | 16.790   | 0.017  | 20.7                  | 22.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1598.5     | 1184.099 | 1.184  | 1597.9                | 1598.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1692.1     | 1253.395 | 1.253  | 1691.4                | 1692.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.3      | 423.938  | 0.424  | 564.9                 | 572.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.593  | 0.278  | 374.6                 | 374.8                    |

Final op time: 1.253 us (1692.1 cycles), bucket=TRISC_0 compute

