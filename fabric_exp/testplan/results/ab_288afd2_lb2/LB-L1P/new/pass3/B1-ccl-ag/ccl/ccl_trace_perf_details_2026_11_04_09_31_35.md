# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_31_35
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

Report: generated/profiler/reports/2026_11_04_09_31_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3829.2     | 2836.420 | 2.836  | 2485.0     | 3813.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2280.6     | 1689.358 | 1.689  | 2201.0     | 2276.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3781.8     | 2801.309 | 2.801  | 2506.0     | 3801.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3862.0     | 2860.741 | 2.861  | 2596.0     | 3847.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3771.2     | 2793.481 | 2.793  | 2456.0     | 3730.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2188.3     | 1620.988 | 1.621  | 2137.0     | 2189.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3815.7     | 2826.444 | 2.826  | 2480.0     | 3980.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3740.0     | 2770.395 | 2.770  | 2468.0     | 3743.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2201.3     | 1630.617 | 1.631  | 2150.0     | 2191.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3796.8     | 2812.444 | 2.812  | 2518.0     | 3782.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2237.6     | 1657.506 | 1.658  | 2179.0     | 2229.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3727.3     | 2760.938 | 2.761  | 2505.0     | 3714.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3825.9     | 2834.012 | 2.834  | 3796.8                | 3822.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3007.4     | 2227.685 | 2.228  | 2237.6                | 3010.3                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2974.7     | 2203.463 | 2.203  | 2188.3                | 2964.3                   |

Final op time: 2.834 us (3825.9 cycles), bucket=NCRISC gather

