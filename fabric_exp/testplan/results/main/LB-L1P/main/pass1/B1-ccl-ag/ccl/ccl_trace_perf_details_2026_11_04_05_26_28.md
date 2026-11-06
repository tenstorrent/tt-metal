# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_26_28
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

Report: generated/profiler/reports/2026_11_04_05_26_43/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3543.4     | 2624.741 | 2.625  | 2360.0     | 3494.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1862.3     | 1379.457 | 1.379  | 1785.0     | 1832.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3473.1     | 2572.667 | 2.573  | 2854.0     | 3449.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3550.5     | 2629.975 | 2.630  | 2873.0     | 3548.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3481.3     | 2578.765 | 2.579  | 2319.0     | 3428.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1891.1     | 1400.840 | 1.401  | 1786.0     | 1879.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3511.1     | 2600.840 | 2.601  | 2547.0     | 3482.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3415.0     | 2529.605 | 2.530  | 2718.0     | 3332.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1944.7     | 1440.494 | 1.440  | 1836.0     | 1924.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3498.8     | 2591.679 | 2.592  | 2766.0     | 3488.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1950.3     | 1444.642 | 1.445  | 1838.0     | 1938.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3458.0     | 2561.457 | 2.561  | 2141.0     | 3462.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3525.9     | 2611.809 | 2.612  | 3498.8                | 3527.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2677.2     | 1983.117 | 1.983  | 1862.3                | 2682.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2691.7     | 1993.864 | 1.994  | 1891.1                | 2701.3                   |

Final op time: 2.612 us (3525.9 cycles), bucket=NCRISC gather

