# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_56_08
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

Report: generated/profiler/reports/2026_11_03_22_56_18/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3573.8     | 2647.235 | 2.647  | 2277.0     | 3574.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1873.5     | 1387.778 | 1.388  | 1776.0     | 1835.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3486.1     | 2582.296 | 2.582  | 2132.0     | 3563.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3592.8     | 2661.309 | 2.661  | 2223.0     | 3582.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3522.4     | 2609.210 | 2.609  | 2166.0     | 3490.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1885.1     | 1396.370 | 1.396  | 1807.0     | 1879.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3535.1     | 2618.568 | 2.619  | 2153.0     | 3472.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3437.1     | 2545.975 | 2.546  | 2190.0     | 3476.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1964.4     | 1455.111 | 1.455  | 1793.0     | 1952.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3515.8     | 2604.272 | 2.604  | 2259.0     | 3472.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1934.2     | 1432.741 | 1.433  | 1836.0     | 1933.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3468.7     | 2569.407 | 2.569  | 2058.0     | 3474.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3554.3     | 2632.846 | 2.633  | 3515.8                | 3554.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2691.8     | 1993.926 | 1.994  | 1873.5                | 2685.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2701.1     | 2000.796 | 2.001  | 1885.1                | 2716.6                   |

Final op time: 2.633 us (3554.3 cycles), bucket=NCRISC gather

