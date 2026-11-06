# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_39_07
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_04_05_39_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1793.8     | 1328.765 | 1.329  | 1709.0     | 1794.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1777.8     | 1316.864 | 1.317  | 1690.0     | 1799.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1828.1     | 1354.148 | 1.354  | 1654.0     | 1837.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1919.6     | 1421.926 | 1.422  | 1742.0     | 1929.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.4      | 423.235  | 0.423  | 407.0      | 575.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.605  | 0.274  | 362.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1935.4     | 1433.605 | 1.434  | 1803.0     | 1943.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1744.2     | 1292.025 | 1.292  | 1634.0     | 1735.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1804.1     | 1336.395 | 1.336  | 1543.0     | 1796.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1897.4     | 1405.506 | 1.406  | 1634.0     | 1909.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 575.0      | 425.926  | 0.426  | 457.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 367.5      | 272.222  | 0.272  | 357.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1864.6     | 1381.185 | 1.381  | 1793.8                | 1864.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1761.0     | 1304.444 | 1.304  | 1744.2                | 1761.0                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1816.1     | 1345.272 | 1.345  | 1804.1                | 1816.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1908.5     | 1413.716 | 1.414  | 1897.4                | 1908.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.2      | 424.580  | 0.425  | 571.4                 | 573.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.4      | 272.914  | 0.273  | 367.5                 | 368.4                    |

Final op time: 1.414 us (1908.5 cycles), bucket=TRISC_0 compute

