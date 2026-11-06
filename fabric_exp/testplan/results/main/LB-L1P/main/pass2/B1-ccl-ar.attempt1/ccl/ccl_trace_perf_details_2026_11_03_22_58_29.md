# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_58_29
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

Report: generated/profiler/reports/2026_11_03_22_58_43/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1638.9     | 1214.000 | 1.214  | 1607.0     | 1623.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.5       | 18.914   | 0.019  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1607.3     | 1190.617 | 1.191  | 1208.0     | 1592.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1698.9     | 1258.444 | 1.258  | 1306.0     | 1684.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.6      | 424.173  | 0.424  | 446.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1716.7     | 1271.654 | 1.272  | 1658.0     | 1686.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.469   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1629.5     | 1207.012 | 1.207  | 1415.0     | 1580.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1720.0     | 1274.074 | 1.274  | 1503.0     | 1677.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.8      | 423.580  | 0.424  | 431.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1677.8     | 1242.827 | 1.243  | 1638.9                | 1677.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.5       | 16.691   | 0.017  | 19.5                  | 22.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1618.4     | 1198.815 | 1.199  | 1607.3                | 1618.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1709.5     | 1266.259 | 1.266  | 1698.9                | 1709.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.2      | 423.877  | 0.424  | 571.8                 | 572.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.877  | 0.274  | 369.7                 | 369.7                    |

Final op time: 1.266 us (1709.5 cycles), bucket=TRISC_0 compute

