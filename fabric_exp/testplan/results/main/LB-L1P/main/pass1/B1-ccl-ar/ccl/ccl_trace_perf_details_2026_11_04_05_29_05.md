# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_29_05
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_04_05_29_15/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1643.6     | 1217.506 | 1.218  | 1591.0     | 1636.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.7       | 19.753   | 0.020  | 18.0       | 25.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1652.8     | 1224.321 | 1.224  | 1479.0     | 1645.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1746.1     | 1293.407 | 1.293  | 1567.0     | 1742.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.2      | 424.593  | 0.425  | 457.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.802  | 0.274  | 366.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1752.4     | 1298.049 | 1.298  | 1655.0     | 1732.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.4       | 14.395   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1640.7     | 1215.358 | 1.215  | 1500.0     | 1633.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1731.8     | 1282.790 | 1.283  | 1591.0     | 1717.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.7      | 422.741  | 0.423  | 432.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.5      | 272.938  | 0.273  | 364.0      | 368.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1698.0     | 1257.778 | 1.258  | 1643.6                | 1698.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.1       | 17.074   | 0.017  | 19.4                  | 23.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1646.8     | 1219.840 | 1.220  | 1640.7                | 1646.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1738.9     | 1288.099 | 1.288  | 1731.8                | 1738.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.0      | 423.667  | 0.424  | 570.7                 | 572.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.0      | 273.370  | 0.273  | 368.5                 | 369.0                    |

Final op time: 1.288 us (1738.9 cycles), bucket=TRISC_0 compute

