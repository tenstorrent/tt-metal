# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_26_47
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

Report: generated/profiler/reports/2026_11_04_09_27_07/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4002.9     | 2965.086 | 2.965  | 2611.0     | 3923.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2296.8     | 1701.309 | 1.701  | 2173.0     | 2293.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3953.1     | 2928.247 | 2.928  | 3270.0     | 3963.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4040.2     | 2992.716 | 2.993  | 3214.0     | 4003.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3927.9     | 2909.580 | 2.910  | 2467.0     | 3847.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2207.0     | 1634.840 | 1.635  | 2126.0     | 2218.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3989.6     | 2955.259 | 2.955  | 2656.0     | 3947.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3897.3     | 2886.864 | 2.887  | 3329.0     | 3868.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2227.5     | 1650.025 | 1.650  | 2117.0     | 2209.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3966.2     | 2937.926 | 2.938  | 3252.0     | 3967.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2257.2     | 1671.975 | 1.672  | 2178.0     | 2250.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3902.1     | 2890.444 | 2.890  | 2487.0     | 3891.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3999.7     | 2962.747 | 2.963  | 3966.2                | 3996.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3094.8     | 2292.432 | 2.292  | 2257.2                | 3097.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3072.5     | 2275.889 | 2.276  | 2207.0                | 3064.8                   |

Final op time: 2.963 us (3999.7 cycles), bucket=NCRISC gather

