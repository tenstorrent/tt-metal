# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_26_49
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

Report: generated/profiler/reports/2026_11_04_09_27_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3454.3     | 2558.741 | 2.559  | 2071.0     | 3413.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1847.8     | 1368.741 | 1.369  | 1780.0     | 1827.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3388.3     | 2509.827 | 2.510  | 2156.0     | 3356.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3471.0     | 2571.086 | 2.571  | 2050.0     | 3401.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3385.4     | 2507.728 | 2.508  | 2010.0     | 3372.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1888.5     | 1398.864 | 1.399  | 1816.0     | 1887.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3467.6     | 2568.617 | 2.569  | 2161.0     | 3756.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3361.3     | 2489.852 | 2.490  | 1968.0     | 3331.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1960.0     | 1451.852 | 1.452  | 1825.0     | 1947.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3441.7     | 2549.383 | 2.549  | 2086.0     | 3403.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1803.5     | 1335.951 | 1.336  | 1748.0     | 1800.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3402.8     | 2520.568 | 2.521  | 2058.0     | 3392.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3458.6     | 2561.957 | 2.562  | 3441.7                | 3461.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2599.5     | 1925.568 | 1.926  | 1803.5                | 2604.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2659.9     | 1970.278 | 1.970  | 1888.5                | 2674.1                   |

Final op time: 2.562 us (3458.6 cycles), bucket=NCRISC gather

