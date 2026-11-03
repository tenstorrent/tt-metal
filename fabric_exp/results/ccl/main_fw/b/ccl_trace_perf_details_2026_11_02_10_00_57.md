# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_00_57
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

Report: generated/profiler/reports/2026_11_02_10_01_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1621.8     | 1201.333 | 1.201  | 1572.0     | 1612.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.6       | 18.198   | 0.018  | 18.0       | 20.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1596.8     | 1182.840 | 1.183  | 1417.0     | 1582.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1690.6     | 1252.296 | 1.252  | 1523.0     | 1670.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 578.3      | 428.346  | 0.428  | 432.0      | 586.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.630  | 0.278  | 366.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1698.3     | 1257.975 | 1.258  | 1650.0     | 1681.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.6       | 15.284   | 0.015  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1587.9     | 1176.222 | 1.176  | 1394.0     | 1577.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1682.9     | 1246.593 | 1.247  | 1483.0     | 1680.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.0      | 423.728  | 0.424  | 436.0      | 583.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.802  | 0.278  | 369.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1660.0     | 1229.654 | 1.230  | 1621.8                | 1660.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.6       | 16.741   | 0.017  | 20.6                  | 22.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1592.4     | 1179.531 | 1.180  | 1587.9                | 1592.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1686.8     | 1249.444 | 1.249  | 1682.9                | 1686.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 575.1      | 426.037  | 0.426  | 572.0                 | 575.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.716  | 0.278  | 374.8                 | 374.9                    |

Final op time: 1.249 us (1686.8 cycles), bucket=TRISC_0 compute

