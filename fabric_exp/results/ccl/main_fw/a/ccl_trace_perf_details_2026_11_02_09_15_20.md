# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_15_20
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

Report: generated/profiler/reports/2026_11_02_09_15_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1756.8     | 1301.309 | 1.301  | 1707.0     | 1752.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1746.8     | 1293.926 | 1.294  | 1669.0     | 1744.0        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1768.2     | 1309.802 | 1.310  | 1578.0     | 1816.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1857.7     | 1376.099 | 1.376  | 1659.0     | 1897.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.9      | 425.086  | 0.425  | 424.0      | 576.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 373.2      | 276.420  | 0.276  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1884.6     | 1396.025 | 1.396  | 1787.0     | 1864.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1745.2     | 1292.765 | 1.293  | 1651.0     | 1747.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1748.5     | 1295.210 | 1.295  | 1527.0     | 1728.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1841.2     | 1363.877 | 1.364  | 1621.0     | 1820.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 565.5      | 418.889  | 0.419  | 426.0      | 571.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.4      | 278.074  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1820.7     | 1348.667 | 1.349  | 1756.8                | 1820.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1746.0     | 1293.346 | 1.293  | 1745.2                | 1746.0                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1758.4     | 1302.506 | 1.303  | 1748.5                | 1758.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1849.5     | 1369.988 | 1.370  | 1841.2                | 1849.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.7      | 421.988  | 0.422  | 565.5                 | 569.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.3      | 277.247  | 0.277  | 373.2                 | 374.3                    |

Final op time: 1.370 us (1849.5 cycles), bucket=TRISC_0 compute

