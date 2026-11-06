# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_26_43
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

Report: generated/profiler/reports/2026_11_04_05_26_58/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3553.0     | 2631.877 | 2.632  | 2137.0     | 3540.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1881.9     | 1393.975 | 1.394  | 1777.0     | 1820.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3486.4     | 2582.519 | 2.583  | 2178.0     | 3408.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3589.0     | 2658.494 | 2.658  | 2214.0     | 3596.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3509.9     | 2599.901 | 2.600  | 2065.0     | 3494.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1881.8     | 1393.901 | 1.394  | 1803.0     | 1875.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3541.7     | 2623.481 | 2.623  | 2153.0     | 3777.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3446.1     | 2552.667 | 2.553  | 2063.0     | 3324.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1982.0     | 1468.123 | 1.468  | 1816.0     | 1954.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3511.0     | 2600.741 | 2.601  | 2145.0     | 3479.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1919.8     | 1422.074 | 1.422  | 1829.0     | 1912.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3471.7     | 2571.605 | 2.572  | 2079.0     | 3466.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3548.7     | 2628.648 | 2.629  | 3511.0                | 3547.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2689.4     | 1992.154 | 1.992  | 1881.9                | 2682.9                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2705.4     | 2004.037 | 2.004  | 1881.8                | 2726.8                   |

Final op time: 2.629 us (3548.7 cycles), bucket=NCRISC gather

