# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_32_52
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

Report: generated/profiler/reports/2026_11_04_09_33_08/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1643.8     | 1217.630 | 1.218  | 1591.0     | 1637.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.6       | 18.938   | 0.019  | 18.0       | 24.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1598.1     | 1183.753 | 1.184  | 1515.0     | 1597.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1732.7     | 1283.506 | 1.284  | 1655.0     | 1729.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.0      | 424.444  | 0.424  | 440.0      | 576.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.704  | 0.274  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1725.7     | 1278.296 | 1.278  | 1693.0     | 1726.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.5       | 13.728   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1583.8     | 1173.210 | 1.173  | 1419.0     | 1572.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1715.3     | 1270.593 | 1.271  | 1538.0     | 1708.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.6      | 421.951  | 0.422  | 417.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.407  | 0.273  | 360.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1684.8     | 1247.963 | 1.248  | 1643.8                | 1684.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.1       | 16.333   | 0.016  | 18.5                  | 22.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1590.9     | 1178.481 | 1.178  | 1583.8                | 1590.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1724.0     | 1277.049 | 1.277  | 1715.3                | 1724.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.3      | 423.198  | 0.423  | 569.6                 | 571.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.556  | 0.274  | 369.1                 | 369.3                    |

Final op time: 1.277 us (1724.0 cycles), bucket=TRISC_0 compute

