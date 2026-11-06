# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_31_07
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

Report: generated/profiler/reports/2026_11_04_09_31_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3360.3     | 2489.086 | 2.489  | 2085.0     | 3370.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1814.4     | 1343.975 | 1.344  | 1772.0     | 1805.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3291.3     | 2438.025 | 2.438  | 2069.0     | 3262.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3379.0     | 2502.988 | 2.503  | 2048.0     | 3284.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3306.0     | 2448.889 | 2.449  | 2021.0     | 3271.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1865.6     | 1381.926 | 1.382  | 1816.0     | 1868.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3354.5     | 2484.815 | 2.485  | 2139.0     | 3507.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3254.1     | 2410.420 | 2.410  | 2010.0     | 3215.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1899.4     | 1406.963 | 1.407  | 1827.0     | 1876.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3331.8     | 2467.975 | 2.468  | 2102.0     | 3295.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1780.2     | 1318.642 | 1.319  | 1744.0     | 1775.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3292.2     | 2438.667 | 2.439  | 2059.0     | 3289.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3356.4     | 2486.216 | 2.486  | 3331.8                | 3357.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2538.6     | 1880.481 | 1.880  | 1780.2                | 2534.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2587.1     | 1916.395 | 1.916  | 1865.6                | 2595.4                   |

Final op time: 2.486 us (3356.4 cycles), bucket=NCRISC gather

