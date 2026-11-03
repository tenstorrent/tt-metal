# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_08_37_25
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

Report: generated/profiler/reports/2026_11_02_08_37_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4052.7     | 3002.000 | 3.002  | 2670.0     | 4017.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2212.9     | 1639.210 | 1.639  | 2112.0     | 2190.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3977.4     | 2946.247 | 2.946  | 2663.0     | 3973.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4061.0     | 3008.148 | 3.008  | 2725.0     | 4025.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3988.7     | 2954.593 | 2.955  | 2612.0     | 3929.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2214.9     | 1640.642 | 1.641  | 2150.0     | 2202.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 4037.8     | 2990.938 | 2.991  | 2675.0     | 4123.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3932.9     | 2913.235 | 2.913  | 2731.0     | 3981.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2288.3     | 1695.062 | 1.695  | 2126.0     | 2282.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 4019.1     | 2977.136 | 2.977  | 2600.0     | 4074.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2261.9     | 1675.481 | 1.675  | 2175.0     | 2259.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3977.6     | 2946.395 | 2.946  | 2532.0     | 3942.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 4042.7     | 2994.556 | 2.995  | 4019.1                | 4045.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3099.1     | 2295.630 | 2.296  | 2212.9                | 3097.4                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3114.6     | 2307.086 | 2.307  | 2214.9                | 3132.9                   |

Final op time: 2.995 us (4042.7 cycles), bucket=NCRISC gather

