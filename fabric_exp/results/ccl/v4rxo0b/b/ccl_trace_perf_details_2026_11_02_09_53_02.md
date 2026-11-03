# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_53_02
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

Report: generated/profiler/reports/2026_11_02_09_53_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3409.4     | 2525.457 | 2.525  | 2098.0     | 3382.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1813.6     | 1343.407 | 1.343  | 1746.0     | 1793.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3341.0     | 2474.840 | 2.475  | 2015.0     | 3447.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3432.4     | 2542.519 | 2.543  | 2068.0     | 3578.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3367.1     | 2494.148 | 2.494  | 2004.0     | 3320.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1847.9     | 1368.790 | 1.369  | 1802.0     | 1845.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3417.6     | 2531.556 | 2.532  | 2073.0     | 3780.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3302.5     | 2446.321 | 2.446  | 1982.0     | 3312.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1923.8     | 1425.062 | 1.425  | 1818.0     | 1912.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3390.0     | 2511.086 | 2.511  | 2103.0     | 3384.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1877.3     | 1390.568 | 1.391  | 1824.0     | 1876.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3352.3     | 2483.210 | 2.483  | 2031.0     | 3337.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3412.3     | 2527.654 | 2.528  | 3390.0                | 3413.5                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2590.1     | 1918.611 | 1.919  | 1813.6                | 2589.9                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2616.3     | 1937.975 | 1.938  | 1847.9                | 2632.4                   |

Final op time: 2.528 us (3412.3 cycles), bucket=NCRISC gather

