# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_55_57
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_03_22_56_07/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4061.2     | 3008.296 | 3.008  | 2583.0     | 3991.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2198.2     | 1628.321 | 1.628  | 2092.0     | 2160.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3973.5     | 2943.309 | 2.943  | 2731.0     | 4003.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4087.0     | 3027.432 | 3.027  | 2717.0     | 4087.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 4020.7     | 2978.296 | 2.978  | 2603.0     | 3946.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2205.8     | 1633.951 | 1.634  | 2120.0     | 2196.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 4019.1     | 2977.111 | 2.977  | 2581.0     | 3984.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3912.9     | 2898.444 | 2.898  | 2728.0     | 3822.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2272.3     | 1683.185 | 1.683  | 2132.0     | 2257.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3987.8     | 2953.901 | 2.954  | 2682.0     | 3915.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2257.0     | 1671.877 | 1.672  | 2166.0     | 2252.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3953.1     | 2928.222 | 2.928  | 2551.0     | 3898.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 4038.8     | 2991.685 | 2.992  | 3987.8                | 4040.1                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3097.2     | 2294.235 | 2.294  | 2198.2                | 3085.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3101.2     | 2297.167 | 2.297  | 2205.8                | 3112.7                   |

Final op time: 2.992 us (4038.8 cycles), bucket=NCRISC gather

