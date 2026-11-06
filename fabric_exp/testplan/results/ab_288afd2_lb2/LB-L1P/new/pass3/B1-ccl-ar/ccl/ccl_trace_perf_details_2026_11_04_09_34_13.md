# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_34_13
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

Report: generated/profiler/reports/2026_11_04_09_34_31/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1740.4     | 1289.210 | 1.289  | 1707.0     | 1734.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.4       | 14.370   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1593.4     | 1180.296 | 1.180  | 1504.0     | 1603.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1724.5     | 1277.407 | 1.277  | 1649.0     | 1730.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.1      | 424.494  | 0.424  | 457.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.580  | 0.274  | 364.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1675.3     | 1240.938 | 1.241  | 1613.0     | 1676.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 32.0       | 23.704   | 0.024  | 18.0       | 30.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1641.0     | 1215.531 | 1.216  | 1507.0     | 1610.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1774.5     | 1314.420 | 1.314  | 1663.0     | 1750.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 575.5      | 426.296  | 0.426  | 410.0      | 581.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.235  | 0.273  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1707.8     | 1265.074 | 1.265  | 1675.3                | 1707.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.7       | 19.037   | 0.019  | 19.4                  | 25.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1617.2     | 1197.914 | 1.198  | 1593.4                | 1617.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1749.5     | 1295.914 | 1.296  | 1724.5                | 1749.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.3      | 425.395  | 0.425  | 573.1                 | 574.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.1      | 273.407  | 0.273  | 368.9                 | 369.1                    |

Final op time: 1.296 us (1749.5 cycles), bucket=TRISC_0 compute

