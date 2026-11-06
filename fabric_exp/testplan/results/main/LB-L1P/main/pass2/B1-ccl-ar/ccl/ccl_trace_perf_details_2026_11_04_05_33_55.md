# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_33_55
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

Report: generated/profiler/reports/2026_11_04_05_34_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1644.4     | 1218.074 | 1.218  | 1591.0     | 1626.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.6       | 18.988   | 0.019  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1633.9     | 1210.296 | 1.210  | 1270.0     | 1641.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1727.1     | 1279.309 | 1.279  | 1357.0     | 1739.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.6      | 425.630  | 0.426  | 470.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.704  | 0.274  | 359.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1729.5     | 1281.086 | 1.281  | 1668.0     | 1698.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.7       | 13.877   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1625.0     | 1203.728 | 1.204  | 1444.0     | 1603.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1717.6     | 1272.321 | 1.272  | 1536.0     | 1700.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.2      | 423.086  | 0.423  | 437.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.802  | 0.274  | 362.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1686.9     | 1249.580 | 1.250  | 1644.4                | 1686.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.2       | 16.432   | 0.016  | 18.7                  | 22.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1629.5     | 1207.012 | 1.207  | 1625.0                | 1629.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1722.3     | 1275.815 | 1.276  | 1717.6                | 1722.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.9      | 424.358  | 0.424  | 571.2                 | 572.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.753  | 0.274  | 369.5                 | 369.6                    |

Final op time: 1.276 us (1722.3 cycles), bucket=TRISC_0 compute

