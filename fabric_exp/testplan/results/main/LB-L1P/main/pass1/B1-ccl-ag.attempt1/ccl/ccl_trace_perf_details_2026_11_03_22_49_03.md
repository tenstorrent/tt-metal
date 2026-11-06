# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_49_03
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

Report: generated/profiler/reports/2026_11_03_22_49_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3552.8     | 2631.704 | 2.632  | 2091.0     | 3528.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1875.2     | 1389.037 | 1.389  | 1781.0     | 1844.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3486.1     | 2582.272 | 2.582  | 2101.0     | 3510.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3575.8     | 2648.765 | 2.649  | 2154.0     | 3540.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3497.0     | 2590.346 | 2.590  | 2013.0     | 3503.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1883.5     | 1395.160 | 1.395  | 1808.0     | 1881.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3548.6     | 2628.568 | 2.629  | 2057.0     | 3766.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3445.6     | 2552.296 | 2.552  | 2093.0     | 3493.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1992.1     | 1475.630 | 1.476  | 1799.0     | 1949.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3516.3     | 2604.667 | 2.605  | 2126.0     | 3629.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1929.9     | 1429.556 | 1.430  | 1826.0     | 1924.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3481.2     | 2578.642 | 2.579  | 2010.0     | 3511.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3548.4     | 2628.426 | 2.628  | 3516.3                | 3550.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2686.9     | 1990.309 | 1.990  | 1875.2                | 2687.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2710.7     | 2007.926 | 2.008  | 1883.5                | 2736.6                   |

Final op time: 2.628 us (3548.4 cycles), bucket=NCRISC gather

