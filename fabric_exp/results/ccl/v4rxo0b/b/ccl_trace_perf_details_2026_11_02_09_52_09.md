# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_52_09
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

Report: generated/profiler/reports/2026_11_02_09_52_21/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3871.0     | 2867.407 | 2.867  | 2725.0     | 3833.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2168.0     | 1605.951 | 1.606  | 2110.0     | 2155.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3790.3     | 2807.605 | 2.808  | 2831.0     | 3752.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3899.5     | 2888.519 | 2.889  | 2820.0     | 3807.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3844.9     | 2848.074 | 2.848  | 2666.0     | 3763.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2161.4     | 1601.012 | 1.601  | 2112.0     | 2166.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3836.9     | 2842.173 | 2.842  | 2774.0     | 3786.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3721.9     | 2756.988 | 2.757  | 2693.0     | 3674.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2247.4     | 1664.765 | 1.665  | 2125.0     | 2230.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3805.2     | 2818.642 | 2.819  | 2584.0     | 3733.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2232.2     | 1653.481 | 1.653  | 2181.0     | 2236.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3775.0     | 2796.272 | 2.796  | 2534.0     | 3791.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3853.1     | 2854.185 | 2.854  | 3805.2                | 3854.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2991.8     | 2216.123 | 2.216  | 2168.0                | 2977.1                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2993.5     | 2217.414 | 2.217  | 2161.4                | 3011.2                   |

Final op time: 2.854 us (3853.1 cycles), bucket=NCRISC gather

