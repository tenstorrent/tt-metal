# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_34_22
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_34_42/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1856.8     | 1375.407 | 1.375  | 1827.0     | 1842.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.0       | 17.753   | 0.018  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1798.2     | 1331.975 | 1.332  | 1448.0     | 1810.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1930.7     | 1430.148 | 1.430  | 1587.0     | 1949.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 566.0      | 419.259  | 0.419  | 412.0      | 570.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.679  | 0.278  | 370.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1935.6     | 1433.778 | 1.434  | 1903.0     | 1931.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.420   | 0.014  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1793.4     | 1328.420 | 1.328  | 1620.0     | 1761.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1927.3     | 1427.605 | 1.428  | 1750.0     | 1892.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.9      | 415.506  | 0.416  | 385.0      | 565.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.506  | 0.278  | 369.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1896.2     | 1404.593 | 1.405  | 1856.8                | 1896.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.7       | 16.086   | 0.016  | 19.5                  | 21.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1795.8     | 1330.198 | 1.330  | 1793.4                | 1795.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1929.0     | 1428.877 | 1.429  | 1927.3                | 1929.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 563.5      | 417.383  | 0.417  | 560.9                 | 563.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.593  | 0.278  | 374.6                 | 374.8                    |

Final op time: 1.429 us (1929.0 cycles), bucket=TRISC_0 compute

