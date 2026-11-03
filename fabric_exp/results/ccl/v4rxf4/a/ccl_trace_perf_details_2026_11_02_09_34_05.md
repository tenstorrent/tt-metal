# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_34_05
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_34_22/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3401.5     | 2519.654 | 2.520  | 2113.0     | 3372.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1819.4     | 1347.728 | 1.348  | 1751.0     | 1792.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3333.9     | 2469.531 | 2.470  | 2125.0     | 3336.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3419.0     | 2532.617 | 2.533  | 2150.0     | 3453.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3348.4     | 2480.272 | 2.480  | 2008.0     | 3282.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1846.2     | 1367.556 | 1.368  | 1790.0     | 1846.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3404.1     | 2521.531 | 2.522  | 1997.0     | 3636.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3295.0     | 2440.741 | 2.441  | 2067.0     | 3271.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1913.3     | 1417.259 | 1.417  | 1818.0     | 1913.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3383.2     | 2506.049 | 2.506  | 2198.0     | 3381.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1870.8     | 1385.802 | 1.386  | 1813.0     | 1870.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3348.6     | 2480.444 | 2.480  | 2053.0     | 3369.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3401.9     | 2519.963 | 2.520  | 3383.2                | 3402.8                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2583.4     | 1913.636 | 1.914  | 1819.4                | 2582.9                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2610.5     | 1933.698 | 1.934  | 1846.2                | 2623.6                   |

Final op time: 2.520 us (3401.9 cycles), bucket=NCRISC gather

