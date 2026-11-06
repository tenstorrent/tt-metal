# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_33_24
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

Report: generated/profiler/reports/2026_11_04_05_33_34/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1651.5     | 1223.309 | 1.223  | 1593.0     | 1643.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.4       | 21.012   | 0.021  | 18.0       | 28.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1655.9     | 1226.593 | 1.227  | 1403.0     | 1709.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1791.9     | 1327.309 | 1.327  | 1553.0     | 1838.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.0      | 423.728  | 0.424  | 476.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1779.0     | 1317.778 | 1.318  | 1652.0     | 1760.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.4       | 14.346   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1627.3     | 1205.432 | 1.205  | 1447.0     | 1640.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1759.8     | 1303.556 | 1.304  | 1576.0     | 1766.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.0      | 423.704  | 0.424  | 434.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 362.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1715.2     | 1270.543 | 1.271  | 1651.5                | 1715.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.9       | 17.679   | 0.018  | 19.4                  | 23.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1641.6     | 1216.012 | 1.216  | 1627.3                | 1641.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1775.8     | 1315.432 | 1.315  | 1759.8                | 1775.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.0      | 423.716  | 0.424  | 572.0                 | 572.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.469  | 0.273  | 369.1                 | 369.2                    |

Final op time: 1.315 us (1775.8 cycles), bucket=TRISC_0 compute

