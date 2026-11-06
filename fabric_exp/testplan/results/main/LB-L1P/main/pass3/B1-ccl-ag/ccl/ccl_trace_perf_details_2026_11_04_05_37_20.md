# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_37_20
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

Report: generated/profiler/reports/2026_11_04_05_37_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4061.9     | 3008.815 | 3.009  | 2733.0     | 3940.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2193.5     | 1624.815 | 1.625  | 2100.0     | 2168.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3970.5     | 2941.136 | 2.941  | 2676.0     | 3960.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4072.9     | 3016.938 | 3.017  | 2557.0     | 4085.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3991.2     | 2956.469 | 2.956  | 2715.0     | 3937.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2207.9     | 1635.457 | 1.635  | 2121.0     | 2215.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 4043.3     | 2995.037 | 2.995  | 2769.0     | 4068.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3925.8     | 2907.975 | 2.908  | 2617.0     | 3912.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2309.0     | 1710.370 | 1.710  | 2154.0     | 2309.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 4008.6     | 2969.333 | 2.969  | 2668.0     | 4004.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2277.6     | 1687.086 | 1.687  | 2163.0     | 2279.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3984.5     | 2951.481 | 2.951  | 2556.0     | 3973.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 4046.7     | 2997.531 | 2.998  | 4008.6                | 4052.6                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3097.0     | 2294.086 | 2.294  | 2193.5                | 3101.7                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3118.0     | 2309.611 | 2.310  | 2207.9                | 3139.8                   |

Final op time: 2.998 us (4046.7 cycles), bucket=NCRISC gather

