# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_46_14
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_46_26/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3408.9     | 2525.086 | 2.525  | 2045.0     | 3359.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1836.8     | 1360.568 | 1.361  | 1760.0     | 1803.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3347.1     | 2479.333 | 2.479  | 2159.0     | 3473.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3430.3     | 2540.988 | 2.541  | 2188.0     | 3503.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3361.7     | 2490.148 | 2.490  | 2043.0     | 3318.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1848.6     | 1369.333 | 1.369  | 1788.0     | 1843.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3406.2     | 2523.086 | 2.523  | 2136.0     | 3699.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3290.2     | 2437.210 | 2.437  | 2064.0     | 3235.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1903.0     | 1409.654 | 1.410  | 1803.0     | 1893.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3379.3     | 2503.185 | 2.503  | 2141.0     | 3368.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1870.4     | 1385.457 | 1.385  | 1830.0     | 1870.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3347.2     | 2479.432 | 2.479  | 2033.0     | 3354.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3406.2     | 2523.086 | 2.523  | 3379.3                | 3407.5                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2589.8     | 1918.346 | 1.918  | 1836.8                | 2580.3                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2611.5     | 1934.438 | 1.934  | 1848.6                | 2625.1                   |

Final op time: 2.523 us (3406.2 cycles), bucket=NCRISC gather

