# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_30_02
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

Report: generated/profiler/reports/2026_11_04_09_30_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1797.6     | 1331.556 | 1.332  | 1711.0     | 1837.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.9       | 14.000   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1668.4     | 1235.852 | 1.236  | 1234.0     | 1695.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1763.8     | 1306.494 | 1.306  | 1321.0     | 1800.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.0      | 420.025  | 0.420  | 422.0      | 569.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 274.000  | 0.274  | 365.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1682.6     | 1246.346 | 1.246  | 1623.0     | 1680.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 38.4       | 28.444   | 0.028  | 18.0       | 39.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1745.2     | 1292.716 | 1.293  | 1531.0     | 1729.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1836.5     | 1360.395 | 1.360  | 1626.0     | 1817.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.4      | 425.457  | 0.425  | 444.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.8      | 273.926  | 0.274  | 364.0      | 370.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1740.1     | 1288.951 | 1.289  | 1682.6                | 1740.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 28.6       | 21.222   | 0.021  | 18.9                  | 28.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1706.8     | 1264.284 | 1.264  | 1668.4                | 1706.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1800.2     | 1333.444 | 1.333  | 1763.8                | 1800.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.7      | 422.741  | 0.423  | 567.0                 | 570.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.9      | 273.963  | 0.274  | 369.8                 | 369.9                    |

Final op time: 1.333 us (1800.2 cycles), bucket=TRISC_0 compute

