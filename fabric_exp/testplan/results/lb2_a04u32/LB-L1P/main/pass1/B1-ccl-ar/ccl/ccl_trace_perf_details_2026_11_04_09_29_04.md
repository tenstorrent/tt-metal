# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_29_04
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

Report: generated/profiler/reports/2026_11_04_09_29_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1660.3     | 1229.852 | 1.230  | 1593.0     | 1670.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.3       | 21.728   | 0.022  | 18.0       | 26.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1704.8     | 1262.840 | 1.263  | 1503.0     | 1744.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1797.8     | 1331.728 | 1.332  | 1600.0     | 1836.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.5      | 423.309  | 0.423  | 414.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 365.0      | 368.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1799.4     | 1332.864 | 1.333  | 1643.0     | 1839.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.4       | 14.370   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1677.1     | 1242.272 | 1.242  | 1389.0     | 1707.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1769.0     | 1310.395 | 1.310  | 1484.0     | 1802.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.4      | 425.457  | 0.425  | 451.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1729.8     | 1281.358 | 1.281  | 1660.3                | 1729.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.4       | 18.049   | 0.018  | 19.4                  | 24.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1690.9     | 1252.556 | 1.253  | 1677.1                | 1690.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1783.4     | 1321.062 | 1.321  | 1769.0                | 1783.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.9      | 424.383  | 0.424  | 571.5                 | 572.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.519  | 0.274  | 369.2                 | 369.2                    |

Final op time: 1.321 us (1783.4 cycles), bucket=TRISC_0 compute

