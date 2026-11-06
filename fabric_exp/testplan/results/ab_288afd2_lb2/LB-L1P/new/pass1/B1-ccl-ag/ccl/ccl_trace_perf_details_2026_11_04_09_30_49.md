# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_30_49
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_09_31_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3364.8     | 2492.469 | 2.492  | 2045.0     | 3339.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1825.2     | 1351.975 | 1.352  | 1778.0     | 1802.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3290.5     | 2437.407 | 2.437  | 2037.0     | 3360.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3397.1     | 2516.346 | 2.516  | 2008.0     | 3510.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3322.7     | 2461.259 | 2.461  | 2044.0     | 3299.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1860.3     | 1378.000 | 1.378  | 1798.0     | 1859.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3363.2     | 2491.235 | 2.491  | 2097.0     | 3726.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3263.1     | 2417.086 | 2.417  | 2121.0     | 3274.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1930.2     | 1429.778 | 1.430  | 1827.0     | 1901.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3319.8     | 2459.111 | 2.459  | 1914.0     | 3343.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1779.9     | 1318.420 | 1.318  | 1750.0     | 1777.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3293.9     | 2439.901 | 2.440  | 2015.0     | 3324.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3361.2     | 2489.790 | 2.490  | 3319.8                | 3364.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2547.7     | 1887.185 | 1.887  | 1779.9                | 2544.1                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2593.7     | 1921.272 | 1.921  | 1860.3                | 2610.3                   |

Final op time: 2.490 us (3361.2 cycles), bucket=NCRISC gather

