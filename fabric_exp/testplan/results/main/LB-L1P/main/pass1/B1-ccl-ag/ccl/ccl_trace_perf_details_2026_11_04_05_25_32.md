# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_25_32
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

Report: generated/profiler/reports/2026_11_04_05_25_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4068.4     | 3013.605 | 3.014  | 2742.0     | 4014.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2198.9     | 1628.840 | 1.629  | 2100.0     | 2167.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3974.4     | 2944.025 | 2.944  | 2683.0     | 4023.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4087.0     | 3027.407 | 3.027  | 2857.0     | 4020.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 4020.0     | 2977.753 | 2.978  | 2700.0     | 3947.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2224.7     | 1647.951 | 1.648  | 2139.0     | 2233.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 4060.1     | 3007.481 | 3.007  | 2735.0     | 4162.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3951.5     | 2927.037 | 2.927  | 2586.0     | 3992.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2356.7     | 1745.704 | 1.746  | 2147.0     | 2384.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 4019.1     | 2977.136 | 2.977  | 2846.0     | 4067.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2270.0     | 1681.457 | 1.681  | 2168.0     | 2266.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3978.1     | 2946.741 | 2.947  | 2484.0     | 4034.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 4058.7     | 3006.407 | 3.006  | 4019.1                | 4064.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3110.1     | 2303.772 | 2.304  | 2198.9                | 3110.7                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3133.5     | 2321.105 | 2.321  | 2224.7                | 3165.6                   |

Final op time: 3.006 us (4058.7 cycles), bucket=NCRISC gather

