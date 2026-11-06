# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_19
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

Report: generated/profiler/reports/2026_11_04_09_28_32/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1645.0     | 1218.519 | 1.219  | 1606.0     | 1627.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.0       | 20.025   | 0.020  | 18.0       | 24.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1596.9     | 1182.889 | 1.183  | 1258.0     | 1600.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1734.1     | 1284.543 | 1.285  | 1399.0     | 1735.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.7      | 424.988  | 0.425  | 452.0      | 576.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 273.975  | 0.274  | 366.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1738.7     | 1287.926 | 1.288  | 1708.0     | 1728.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.1       | 14.914   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1599.0     | 1184.420 | 1.184  | 1409.0     | 1598.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1733.4     | 1284.025 | 1.284  | 1529.0     | 1707.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.9      | 424.395  | 0.424  | 456.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.7      | 273.136  | 0.273  | 363.0      | 368.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1691.8     | 1253.222 | 1.253  | 1645.0                | 1691.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.6       | 17.469   | 0.017  | 20.1                  | 23.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1597.9     | 1183.654 | 1.184  | 1596.9                | 1597.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1733.8     | 1284.284 | 1.284  | 1733.4                | 1733.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.3      | 424.691  | 0.425  | 572.9                 | 573.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.556  | 0.274  | 368.7                 | 369.3                    |

Final op time: 1.284 us (1733.8 cycles), bucket=TRISC_0 compute

