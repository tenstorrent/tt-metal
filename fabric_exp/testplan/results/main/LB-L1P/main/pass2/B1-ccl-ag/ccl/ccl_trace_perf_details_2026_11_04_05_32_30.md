# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_32_30
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

Report: generated/profiler/reports/2026_11_04_05_32_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3540.1     | 2622.296 | 2.622  | 2093.0     | 3496.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1868.3     | 1383.926 | 1.384  | 1791.0     | 1845.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3475.7     | 2574.617 | 2.575  | 2085.0     | 3515.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3583.3     | 2654.321 | 2.654  | 2108.0     | 3565.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3498.6     | 2591.556 | 2.592  | 2011.0     | 3491.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1888.7     | 1399.012 | 1.399  | 1791.0     | 1902.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3536.6     | 2619.704 | 2.620  | 2123.0     | 3638.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3429.1     | 2540.099 | 2.540  | 2131.0     | 3486.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1990.3     | 1474.272 | 1.474  | 1824.0     | 1956.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3506.7     | 2597.580 | 2.598  | 2214.0     | 3443.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1942.1     | 1438.568 | 1.439  | 1835.0     | 1932.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3467.6     | 2568.593 | 2.569  | 2020.0     | 3540.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3541.7     | 2623.475 | 2.623  | 3506.7                | 3538.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2684.5     | 1988.537 | 1.989  | 1868.3                | 2685.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2705.6     | 2004.123 | 2.004  | 1888.7                | 2728.9                   |

Final op time: 2.623 us (3541.7 cycles), bucket=NCRISC gather

