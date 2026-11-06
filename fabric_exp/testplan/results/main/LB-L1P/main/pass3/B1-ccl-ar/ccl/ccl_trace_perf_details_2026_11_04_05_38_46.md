# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_38_46
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

Report: generated/profiler/reports/2026_11_04_05_38_56/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1662.3     | 1231.309 | 1.231  | 1589.0     | 1673.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 34.5       | 25.531   | 0.026  | 18.0       | 35.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1699.3     | 1258.716 | 1.259  | 1445.0     | 1721.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1830.6     | 1356.000 | 1.356  | 1594.0     | 1851.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.4      | 426.222  | 0.426  | 455.0      | 577.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1825.2     | 1352.000 | 1.352  | 1657.0     | 1840.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.6       | 14.543   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1661.3     | 1230.568 | 1.231  | 1458.0     | 1674.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1800.7     | 1333.852 | 1.334  | 1590.0     | 1815.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.3      | 425.383  | 0.425  | 452.0      | 579.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.630  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1743.7     | 1291.654 | 1.292  | 1662.3                | 1743.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 27.1       | 20.037   | 0.020  | 19.6                  | 27.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1680.3     | 1244.642 | 1.245  | 1661.3                | 1680.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1815.7     | 1344.926 | 1.345  | 1800.7                | 1815.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.8      | 425.802  | 0.426  | 574.3                 | 574.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.531  | 0.274  | 369.1                 | 369.3                    |

Final op time: 1.345 us (1815.7 cycles), bucket=TRISC_0 compute

