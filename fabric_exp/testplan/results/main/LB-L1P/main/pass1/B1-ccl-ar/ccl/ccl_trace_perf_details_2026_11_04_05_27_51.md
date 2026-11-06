# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_27_51
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_04_05_28_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1646.9     | 1219.926 | 1.220  | 1595.0     | 1627.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 23.8       | 17.605   | 0.018  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1589.5     | 1177.407 | 1.177  | 1194.0     | 1604.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1724.6     | 1277.481 | 1.277  | 1332.0     | 1740.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.0      | 422.938  | 0.423  | 428.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.556  | 0.274  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1732.8     | 1283.531 | 1.284  | 1689.0     | 1727.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.6       | 14.519   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1599.8     | 1185.012 | 1.185  | 1306.0     | 1562.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1736.1     | 1285.975 | 1.286  | 1455.0     | 1686.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.7      | 420.519  | 0.421  | 455.0      | 573.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 360.0      | 370.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1689.8     | 1251.728 | 1.252  | 1646.9                | 1689.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.7       | 16.062   | 0.016  | 19.6                  | 21.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1594.6     | 1181.210 | 1.181  | 1589.5                | 1594.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1730.3     | 1281.728 | 1.282  | 1724.6                | 1730.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.3      | 421.728  | 0.422  | 567.7                 | 569.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.716  | 0.274  | 369.3                 | 369.5                    |

Final op time: 1.282 us (1730.3 cycles), bucket=TRISC_0 compute

