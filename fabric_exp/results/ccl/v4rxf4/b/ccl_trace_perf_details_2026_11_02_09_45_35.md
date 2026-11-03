# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_45_35
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

Report: generated/profiler/reports/2026_11_02_09_45_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3415.0     | 2529.630 | 2.530  | 2068.0     | 3363.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1807.7     | 1339.012 | 1.339  | 1758.0     | 1785.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3348.4     | 2480.272 | 2.480  | 2080.0     | 3413.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3445.8     | 2552.420 | 2.552  | 2097.0     | 3472.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3379.8     | 2503.556 | 2.504  | 2007.0     | 3314.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1843.5     | 1365.556 | 1.366  | 1787.0     | 1838.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3410.5     | 2526.272 | 2.526  | 2000.0     | 3698.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3303.4     | 2446.963 | 2.447  | 2054.0     | 3293.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1933.5     | 1432.222 | 1.432  | 1786.0     | 1924.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3391.1     | 2511.901 | 2.512  | 2156.0     | 3384.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1880.8     | 1393.185 | 1.393  | 1830.0     | 1880.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3357.2     | 2486.840 | 2.487  | 2009.0     | 3338.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3415.6     | 2530.056 | 2.530  | 3391.1                | 3412.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2592.9     | 1920.679 | 1.921  | 1807.7                | 2592.1                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2620.7     | 1941.222 | 1.941  | 1843.5                | 2640.9                   |

Final op time: 2.530 us (3415.6 cycles), bucket=NCRISC gather

