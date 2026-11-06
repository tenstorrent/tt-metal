# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_32_18
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_09_32_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1641.7     | 1216.099 | 1.216  | 1605.0     | 1629.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.4       | 19.556   | 0.020  | 18.0       | 23.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1599.3     | 1184.642 | 1.185  | 1500.0     | 1601.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1733.0     | 1283.704 | 1.284  | 1620.0     | 1734.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.5      | 424.074  | 0.424  | 471.0      | 575.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.309  | 0.273  | 362.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1727.7     | 1279.753 | 1.280  | 1684.0     | 1722.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.420   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1582.7     | 1172.395 | 1.172  | 1430.0     | 1555.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1713.9     | 1269.580 | 1.270  | 1558.0     | 1691.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 575.3      | 426.123  | 0.426  | 455.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.6      | 273.037  | 0.273  | 360.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1684.7     | 1247.926 | 1.248  | 1641.7                | 1684.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.9       | 16.988   | 0.017  | 19.5                  | 22.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1591.0     | 1178.519 | 1.179  | 1582.7                | 1591.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1723.5     | 1276.642 | 1.277  | 1713.9                | 1723.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.9      | 425.099  | 0.425  | 572.5                 | 573.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.8      | 273.173  | 0.273  | 368.6                 | 368.8                    |

Final op time: 1.277 us (1723.5 cycles), bucket=TRISC_0 compute

