# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_38_36
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_05_38_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1644.1     | 1217.877 | 1.218  | 1593.0     | 1630.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.7       | 19.778   | 0.020  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1607.9     | 1191.037 | 1.191  | 1279.0     | 1604.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1746.6     | 1293.802 | 1.294  | 1414.0     | 1746.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.4      | 423.235  | 0.423  | 440.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.630  | 0.274  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1740.5     | 1289.235 | 1.289  | 1644.0     | 1725.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.6       | 13.753   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1599.7     | 1184.963 | 1.185  | 1444.0     | 1592.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1735.3     | 1285.407 | 1.285  | 1587.0     | 1720.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.3      | 423.926  | 0.424  | 455.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.284  | 0.273  | 360.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1692.3     | 1253.556 | 1.254  | 1644.1                | 1692.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.6       | 16.765   | 0.017  | 18.6                  | 22.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1603.8     | 1188.000 | 1.188  | 1599.7                | 1603.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1741.0     | 1289.605 | 1.290  | 1735.3                | 1741.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.8      | 423.580  | 0.424  | 571.4                 | 571.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.457  | 0.273  | 368.9                 | 369.2                    |

Final op time: 1.290 us (1741.0 cycles), bucket=TRISC_0 compute

