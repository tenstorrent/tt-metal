# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_56_18
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_03_22_56_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3576.1     | 2648.988 | 2.649  | 2153.0     | 3623.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1887.8     | 1398.395 | 1.398  | 1777.0     | 1849.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3487.7     | 2583.457 | 2.583  | 2172.0     | 3568.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3595.7     | 2663.481 | 2.663  | 2221.0     | 3635.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3532.4     | 2616.617 | 2.617  | 2047.0     | 3494.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1896.7     | 1404.963 | 1.405  | 1802.0     | 1898.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3549.3     | 2629.086 | 2.629  | 2090.0     | 3691.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3451.5     | 2556.642 | 2.557  | 2041.0     | 3476.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1983.1     | 1468.963 | 1.469  | 1792.0     | 1950.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3520.4     | 2607.704 | 2.608  | 2105.0     | 3592.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1925.6     | 1426.395 | 1.426  | 1840.0     | 1924.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3484.7     | 2581.235 | 2.581  | 2147.0     | 3440.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3560.4     | 2637.315 | 2.637  | 3520.4                | 3562.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2699.3     | 1999.512 | 2.000  | 1887.8                | 2688.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2713.0     | 2009.654 | 2.010  | 1896.7                | 2733.9                   |

Final op time: 2.637 us (3560.4 cycles), bucket=NCRISC gather

