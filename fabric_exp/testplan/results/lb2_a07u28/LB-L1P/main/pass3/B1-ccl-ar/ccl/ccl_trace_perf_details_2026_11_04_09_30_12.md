# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_30_12
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

Report: generated/profiler/reports/2026_11_04_09_30_22/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1749.7     | 1296.049 | 1.296  | 1709.0     | 1728.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.9       | 13.975   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1632.4     | 1209.160 | 1.209  | 1434.0     | 1627.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1725.8     | 1278.395 | 1.278  | 1530.0     | 1719.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.9      | 421.432  | 0.421  | 432.0      | 571.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.0      | 274.049  | 0.274  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1679.7     | 1244.247 | 1.244  | 1598.0     | 1685.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 33.7       | 24.963   | 0.025  | 18.0       | 33.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1684.3     | 1247.630 | 1.248  | 1568.0     | 1643.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1776.4     | 1315.852 | 1.316  | 1658.0     | 1736.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.3      | 424.642  | 0.425  | 470.0      | 573.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 365.0      | 369.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1714.7     | 1270.148 | 1.270  | 1679.7                | 1714.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 26.3       | 19.469   | 0.019  | 18.9                  | 26.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1658.3     | 1228.395 | 1.228  | 1632.4                | 1658.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1751.1     | 1297.123 | 1.297  | 1725.8                | 1751.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.1      | 423.037  | 0.423  | 568.9                 | 571.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.790  | 0.274  | 369.3                 | 369.6                    |

Final op time: 1.297 us (1751.1 cycles), bucket=TRISC_0 compute

