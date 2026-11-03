# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_48_11
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_48_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1630.5     | 1207.778 | 1.208  | 1575.0     | 1615.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.4       | 18.049   | 0.018  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1603.1     | 1187.457 | 1.187  | 1458.0     | 1601.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1695.9     | 1256.198 | 1.256  | 1560.0     | 1696.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.2      | 427.531  | 0.428  | 429.0      | 585.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.7      | 278.272  | 0.278  | 371.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1703.0     | 1261.506 | 1.262  | 1668.0     | 1699.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.3       | 15.012   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1591.5     | 1178.914 | 1.179  | 1445.0     | 1578.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1689.9     | 1251.778 | 1.252  | 1522.0     | 1672.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 565.9      | 419.210  | 0.419  | 434.0      | 576.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.802  | 0.278  | 368.0      | 375.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1666.8     | 1234.642 | 1.235  | 1630.5                | 1666.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.3       | 16.531   | 0.017  | 20.3                  | 22.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1597.3     | 1183.185 | 1.183  | 1591.5                | 1597.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1692.9     | 1253.988 | 1.254  | 1689.9                | 1692.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.5      | 423.370  | 0.423  | 565.9                 | 571.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.4      | 278.037  | 0.278  | 375.0                 | 375.4                    |

Final op time: 1.254 us (1692.9 cycles), bucket=TRISC_0 compute

