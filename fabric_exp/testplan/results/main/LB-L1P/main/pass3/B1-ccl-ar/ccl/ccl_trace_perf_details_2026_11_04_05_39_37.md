# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_39_37
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

Report: generated/profiler/reports/2026_11_04_05_39_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1650.1     | 1222.272 | 1.222  | 1593.0     | 1655.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.9       | 19.926   | 0.020  | 18.0       | 25.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1690.1     | 1251.901 | 1.252  | 1544.0     | 1677.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1781.9     | 1319.901 | 1.320  | 1598.0     | 1783.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.0      | 423.728  | 0.424  | 405.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 365.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1775.3     | 1315.012 | 1.315  | 1663.0     | 1734.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.6       | 14.494   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1660.3     | 1229.827 | 1.230  | 1421.0     | 1660.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1753.0     | 1298.543 | 1.299  | 1488.0     | 1757.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.9      | 424.395  | 0.424  | 431.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.827  | 0.274  | 366.0      | 369.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1712.7     | 1268.642 | 1.269  | 1650.1                | 1712.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.2       | 17.210   | 0.017  | 19.6                  | 23.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1675.2     | 1240.864 | 1.241  | 1660.3                | 1675.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1767.4     | 1309.222 | 1.309  | 1753.0                | 1767.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.5      | 424.062  | 0.424  | 572.0                 | 572.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.852  | 0.274  | 369.7                 | 369.7                    |

Final op time: 1.309 us (1767.4 cycles), bucket=TRISC_0 compute

