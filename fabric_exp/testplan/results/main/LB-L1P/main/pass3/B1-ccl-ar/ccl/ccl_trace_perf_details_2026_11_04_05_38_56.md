# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_38_56
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

Report: generated/profiler/reports/2026_11_04_05_39_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1647.6     | 1220.469 | 1.220  | 1591.0     | 1628.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.2       | 20.889   | 0.021  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1645.9     | 1219.185 | 1.219  | 1333.0     | 1654.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1779.1     | 1317.852 | 1.318  | 1469.0     | 1796.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.5      | 425.531  | 0.426  | 455.0      | 577.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.383  | 0.273  | 360.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1785.0     | 1322.247 | 1.322  | 1697.0     | 1803.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.9       | 13.975   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1623.9     | 1202.914 | 1.203  | 1401.0     | 1624.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1762.6     | 1305.630 | 1.306  | 1525.0     | 1759.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.5      | 425.556  | 0.426  | 466.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.457  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1716.3     | 1271.358 | 1.271  | 1647.6                | 1716.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.432   | 0.017  | 18.9                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1634.9     | 1211.049 | 1.211  | 1623.9                | 1634.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1770.8     | 1311.741 | 1.312  | 1762.6                | 1770.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.5      | 425.543  | 0.426  | 574.5                 | 574.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.1      | 273.420  | 0.273  | 369.1                 | 369.1                    |

Final op time: 1.312 us (1770.8 cycles), bucket=TRISC_0 compute

