# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_48_00
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

Report: generated/profiler/reports/2026_11_03_22_48_18/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4091.0     | 3030.346 | 3.030  | 2758.0     | 3992.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2187.8     | 1620.568 | 1.621  | 2094.0     | 2161.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3989.8     | 2955.432 | 2.955  | 2939.0     | 3993.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4123.4     | 3054.346 | 3.054  | 2990.0     | 4010.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 4046.7     | 2997.531 | 2.998  | 2665.0     | 3938.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2216.7     | 1641.975 | 1.642  | 2132.0     | 2224.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 4062.4     | 3009.185 | 3.009  | 2735.0     | 4178.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3935.0     | 2914.840 | 2.915  | 2758.0     | 3988.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2284.5     | 1692.198 | 1.692  | 2101.0     | 2264.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3998.7     | 2962.000 | 2.962  | 2028.0     | 4059.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2258.5     | 1672.963 | 1.673  | 2177.0     | 2274.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3963.6     | 2936.000 | 2.936  | 2530.0     | 3918.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 4068.9     | 3013.969 | 3.014  | 3998.7                | 4076.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3107.0     | 2301.475 | 2.301  | 2187.8                | 3096.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3113.6     | 2306.401 | 2.306  | 2216.7                | 3124.0                   |

Final op time: 3.014 us (4068.9 cycles), bucket=NCRISC gather

