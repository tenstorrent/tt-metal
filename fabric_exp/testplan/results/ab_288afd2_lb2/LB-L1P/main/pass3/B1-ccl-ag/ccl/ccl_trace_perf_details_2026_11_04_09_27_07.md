# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_07
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_09_27_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3508.4     | 2598.840 | 2.599  | 2015.0     | 3562.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1963.1     | 1454.173 | 1.454  | 1864.0     | 1956.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3448.2     | 2554.222 | 2.554  | 2232.0     | 3361.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3526.8     | 2612.469 | 2.612  | 2283.0     | 3395.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3439.4     | 2547.704 | 2.548  | 2000.0     | 3446.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1884.5     | 1395.926 | 1.396  | 1802.0     | 1874.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3497.2     | 2590.543 | 2.591  | 2119.0     | 3528.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3398.1     | 2517.111 | 2.517  | 2174.0     | 3336.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1888.4     | 1398.790 | 1.399  | 1808.0     | 1879.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3464.8     | 2566.543 | 2.567  | 2262.0     | 3402.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1941.4     | 1438.099 | 1.438  | 1823.0     | 1948.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3419.9     | 2533.259 | 2.533  | 2072.0     | 3434.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3499.3     | 2592.099 | 2.592  | 3464.8                | 3502.8                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2685.5     | 1989.272 | 1.989  | 1941.4                | 2680.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2660.2     | 1970.549 | 1.971  | 1884.5                | 2654.1                   |

Final op time: 2.592 us (3499.3 cycles), bucket=NCRISC gather

