# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_37_51
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_04_05_38_01/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3535.8     | 2619.136 | 2.619  | 2083.0     | 3532.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1877.3     | 1390.593 | 1.391  | 1769.0     | 1857.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3465.3     | 2566.889 | 2.567  | 2151.0     | 3628.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3558.4     | 2635.852 | 2.636  | 2101.0     | 3782.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3486.0     | 2582.222 | 2.582  | 2027.0     | 3482.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1879.1     | 1391.901 | 1.392  | 1799.0     | 1875.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3539.8     | 2622.099 | 2.622  | 2038.0     | 3966.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3440.0     | 2548.123 | 2.548  | 2062.0     | 3455.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1970.4     | 1459.580 | 1.460  | 1812.0     | 1941.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3511.3     | 2600.963 | 2.601  | 2127.0     | 3521.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1931.7     | 1430.864 | 1.431  | 1829.0     | 1917.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3468.0     | 2568.889 | 2.569  | 2038.0     | 3480.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3536.3     | 2619.512 | 2.620  | 3511.3                | 3537.8                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2683.7     | 1987.951 | 1.988  | 1877.3                | 2685.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2695.7     | 1996.815 | 1.997  | 1879.1                | 2717.9                   |

Final op time: 2.620 us (3536.3 cycles), bucket=NCRISC gather

