# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_01_07
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

Report: generated/profiler/reports/2026_11_02_10_01_16/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1646.8     | 1219.852 | 1.220  | 1575.0     | 1663.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.4       | 21.802   | 0.022  | 18.0       | 29.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1630.0     | 1207.407 | 1.207  | 1423.0     | 1634.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1725.5     | 1278.123 | 1.278  | 1503.0     | 1724.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.1      | 427.481  | 0.427  | 412.0      | 585.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.457  | 0.277  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1734.7     | 1284.988 | 1.285  | 1652.0     | 1726.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.9       | 14.741   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1620.0     | 1200.000 | 1.200  | 1434.0     | 1609.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1713.8     | 1269.506 | 1.270  | 1532.0     | 1700.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.7      | 421.259  | 0.421  | 429.0      | 577.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.605  | 0.278  | 369.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1690.8     | 1252.420 | 1.252  | 1646.8                | 1690.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.7       | 18.272   | 0.018  | 19.9                  | 24.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1625.0     | 1203.704 | 1.204  | 1620.0                | 1625.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1719.7     | 1273.815 | 1.274  | 1713.8                | 1719.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.9      | 424.370  | 0.424  | 568.7                 | 572.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.7      | 277.531  | 0.278  | 374.6                 | 374.7                    |

Final op time: 1.274 us (1719.7 cycles), bucket=TRISC_0 compute

