# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_59_35
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

Report: generated/profiler/reports/2026_11_02_09_59_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3569.9     | 2644.346 | 2.644  | 2225.0     | 3424.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1878.2     | 1391.259 | 1.391  | 1764.0     | 1827.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3508.5     | 2598.889 | 2.599  | 2172.0     | 3525.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3606.4     | 2671.407 | 2.671  | 2379.0     | 3557.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3526.6     | 2612.272 | 2.612  | 2147.0     | 3320.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1873.1     | 1387.457 | 1.387  | 1780.0     | 1871.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3558.2     | 2635.704 | 2.636  | 2224.0     | 3466.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3442.7     | 2550.148 | 2.550  | 2017.0     | 3407.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2003.0     | 1483.704 | 1.484  | 1789.0     | 1989.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3530.4     | 2615.136 | 2.615  | 2085.0     | 3498.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1901.8     | 1408.716 | 1.409  | 1830.0     | 1901.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3488.5     | 2584.099 | 2.584  | 2051.0     | 3396.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3566.2     | 2641.648 | 2.642  | 3530.4                | 3564.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2687.3     | 1990.599 | 1.991  | 1878.2                | 2672.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2718.3     | 2013.537 | 2.014  | 1873.1                | 2745.8                   |

Final op time: 2.642 us (3566.2 cycles), bucket=NCRISC gather

