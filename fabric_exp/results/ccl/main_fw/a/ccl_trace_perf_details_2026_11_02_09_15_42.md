# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_15_42
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

Report: generated/profiler/reports/2026_11_02_09_15_52/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1632.9     | 1209.580 | 1.210  | 1575.0     | 1623.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.2       | 19.432   | 0.019  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1617.3     | 1198.000 | 1.198  | 1288.0     | 1629.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1711.5     | 1267.802 | 1.268  | 1406.0     | 1722.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.9      | 428.049  | 0.428  | 390.0      | 593.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.1      | 277.827  | 0.278  | 369.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1720.0     | 1274.074 | 1.274  | 1636.0     | 1726.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.309   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1610.0     | 1192.617 | 1.193  | 1377.0     | 1599.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1707.1     | 1264.519 | 1.265  | 1468.0     | 1699.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 563.5      | 417.407  | 0.417  | 381.0      | 568.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.802  | 0.278  | 370.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1676.5     | 1241.827 | 1.242  | 1632.9                | 1676.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.370   | 0.017  | 20.7                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1613.7     | 1195.309 | 1.195  | 1610.0                | 1613.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1709.3     | 1266.160 | 1.266  | 1707.1                | 1709.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.7      | 422.728  | 0.423  | 563.5                 | 570.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.1      | 277.815  | 0.278  | 375.0                 | 375.1                    |

Final op time: 1.266 us (1709.3 cycles), bucket=TRISC_0 compute

