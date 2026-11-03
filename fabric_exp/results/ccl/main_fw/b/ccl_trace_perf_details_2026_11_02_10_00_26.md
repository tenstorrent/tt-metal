# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_00_26
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_10_00_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1636.0     | 1211.877 | 1.212  | 1585.0     | 1642.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.1       | 20.049   | 0.020  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1596.3     | 1182.444 | 1.182  | 1295.0     | 1598.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1731.1     | 1282.272 | 1.282  | 1429.0     | 1732.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 561.2      | 415.728  | 0.416  | 402.0      | 567.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.630  | 0.278  | 365.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1733.1     | 1283.802 | 1.284  | 1663.0     | 1731.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.4       | 15.877   | 0.016  | 18.0       | 19.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1591.9     | 1179.185 | 1.179  | 1372.0     | 1588.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1729.8     | 1281.309 | 1.281  | 1517.0     | 1727.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 556.2      | 411.975  | 0.412  | 417.0      | 562.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.481  | 0.277  | 370.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1684.6     | 1247.840 | 1.248  | 1636.0                | 1684.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.2       | 17.963   | 0.018  | 21.4                  | 24.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1594.1     | 1180.815 | 1.181  | 1591.9                | 1594.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1730.4     | 1281.790 | 1.282  | 1729.8                | 1730.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 558.7      | 413.852  | 0.414  | 556.2                 | 558.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.7      | 277.556  | 0.278  | 374.6                 | 374.7                    |

Final op time: 1.282 us (1730.4 cycles), bucket=TRISC_0 compute

