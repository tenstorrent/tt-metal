# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_28_44
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

Report: generated/profiler/reports/2026_11_04_05_28_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1646.4     | 1219.580 | 1.220  | 1587.0     | 1644.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.9       | 20.691   | 0.021  | 18.0       | 24.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1689.2     | 1251.259 | 1.251  | 1484.0     | 1720.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1784.5     | 1321.877 | 1.322  | 1592.0     | 1822.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.1      | 425.259  | 0.425  | 425.0      | 575.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.3      | 274.272  | 0.274  | 365.0      | 371.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1792.1     | 1327.481 | 1.327  | 1696.0     | 1839.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.6       | 13.802   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1667.6     | 1235.284 | 1.235  | 1406.0     | 1694.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1762.3     | 1305.432 | 1.305  | 1490.0     | 1790.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.3      | 420.247  | 0.420  | 425.0      | 573.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 273.975  | 0.274  | 364.0      | 370.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1719.3     | 1273.531 | 1.274  | 1646.4                | 1719.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.3       | 17.247   | 0.017  | 18.6                  | 23.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1678.4     | 1243.272 | 1.243  | 1667.6                | 1678.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1773.4     | 1313.654 | 1.314  | 1762.3                | 1773.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.7      | 422.753  | 0.423  | 567.3                 | 570.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 370.1      | 274.123  | 0.274  | 369.9                 | 370.1                    |

Final op time: 1.314 us (1773.4 cycles), bucket=TRISC_0 compute

