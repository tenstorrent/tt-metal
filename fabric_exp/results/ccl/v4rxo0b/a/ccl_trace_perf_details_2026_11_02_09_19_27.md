# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_19_27
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

Report: generated/profiler/reports/2026_11_02_09_19_43/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3425.7     | 2537.531 | 2.538  | 2102.0     | 3407.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1815.6     | 1344.914 | 1.345  | 1739.0     | 1781.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3349.4     | 2481.037 | 2.481  | 2104.0     | 3339.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3449.5     | 2555.160 | 2.555  | 2127.0     | 3429.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3385.9     | 2508.074 | 2.508  | 2045.0     | 3312.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1840.4     | 1363.235 | 1.363  | 1792.0     | 1844.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3422.1     | 2534.889 | 2.535  | 2129.0     | 3632.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3316.1     | 2456.370 | 2.456  | 2115.0     | 3307.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1934.2     | 1432.716 | 1.433  | 1798.0     | 1932.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3390.8     | 2511.704 | 2.512  | 2149.0     | 3399.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1879.8     | 1392.444 | 1.392  | 1819.0     | 1883.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3350.5     | 2481.877 | 2.482  | 2061.0     | 3364.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3422.0     | 2534.821 | 2.535  | 3390.8                | 3423.9                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2599.4     | 1925.451 | 1.925  | 1815.6                | 2597.9                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2618.6     | 1939.716 | 1.940  | 1840.4                | 2641.8                   |

Final op time: 2.535 us (3422.0 cycles), bucket=NCRISC gather

