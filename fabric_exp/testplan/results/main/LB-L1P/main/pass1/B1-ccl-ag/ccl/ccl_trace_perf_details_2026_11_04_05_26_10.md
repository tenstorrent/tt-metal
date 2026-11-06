# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_26_10
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_05_26_28/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3566.6     | 2641.901 | 2.642  | 2214.0     | 3500.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1870.3     | 1385.407 | 1.385  | 1778.0     | 1846.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3504.5     | 2595.951 | 2.596  | 2163.0     | 3506.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3620.0     | 2681.481 | 2.681  | 2205.0     | 3496.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3536.6     | 2619.679 | 2.620  | 2149.0     | 3410.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1893.5     | 1402.617 | 1.403  | 1813.0     | 1889.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3521.4     | 2608.420 | 2.608  | 2278.0     | 3541.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3421.9     | 2534.716 | 2.535  | 2041.0     | 3461.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2003.2     | 1483.827 | 1.484  | 1818.0     | 1990.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3504.3     | 2595.778 | 2.596  | 2245.0     | 3544.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1932.8     | 1431.728 | 1.432  | 1836.0     | 1929.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3459.8     | 2562.815 | 2.563  | 2110.0     | 3526.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3553.1     | 2631.895 | 2.632  | 3504.3                | 3544.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2690.4     | 1992.883 | 1.993  | 1870.3                | 2677.3                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2715.3     | 2011.302 | 2.011  | 1893.5                | 2731.5                   |

Final op time: 2.632 us (3553.1 cycles), bucket=NCRISC gather

