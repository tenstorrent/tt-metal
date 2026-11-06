# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_34_26
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

Report: generated/profiler/reports/2026_11_04_05_34_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1662.6     | 1231.531 | 1.232  | 1589.0     | 1672.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 30.0       | 22.198   | 0.022  | 18.0       | 27.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1668.0     | 1235.556 | 1.236  | 1420.0     | 1695.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1758.6     | 1302.642 | 1.303  | 1522.0     | 1780.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 576.6      | 427.111  | 0.427  | 459.0      | 575.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.457  | 0.273  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1756.9     | 1301.432 | 1.301  | 1670.0     | 1732.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.7       | 13.827   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1646.8     | 1219.877 | 1.220  | 1502.0     | 1629.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1739.4     | 1288.444 | 1.288  | 1578.0     | 1717.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.2      | 424.568  | 0.425  | 442.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.284  | 0.273  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1709.8     | 1266.481 | 1.266  | 1662.6                | 1709.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.3       | 18.012   | 0.018  | 18.7                  | 24.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1657.4     | 1227.716 | 1.228  | 1646.8                | 1657.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1749.0     | 1295.543 | 1.296  | 1739.4                | 1749.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.9      | 425.840  | 0.426  | 573.2                 | 574.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.1      | 273.370  | 0.273  | 368.9                 | 369.1                    |

Final op time: 1.296 us (1749.0 cycles), bucket=TRISC_0 compute

