# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_52_48
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

Report: generated/profiler/reports/2026_11_02_09_53_01/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3448.6     | 2554.519 | 2.555  | 2086.0     | 3375.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1819.6     | 1347.877 | 1.348  | 1732.0     | 1789.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3366.8     | 2493.951 | 2.494  | 2096.0     | 3347.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3476.2     | 2574.938 | 2.575  | 2139.0     | 3459.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3411.1     | 2526.716 | 2.527  | 2029.0     | 3329.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1846.3     | 1367.630 | 1.368  | 1799.0     | 1832.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3435.1     | 2544.519 | 2.545  | 2124.0     | 3758.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3318.3     | 2457.975 | 2.458  | 2071.0     | 3285.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1948.9     | 1443.605 | 1.444  | 1829.0     | 1934.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3389.4     | 2510.667 | 2.511  | 1573.0     | 3417.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1869.6     | 1384.914 | 1.385  | 1817.0     | 1865.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3358.4     | 2487.679 | 2.488  | 1970.0     | 3359.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3437.3     | 2546.160 | 2.546  | 3389.4                | 3441.8                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2604.7     | 1929.370 | 1.929  | 1819.6                | 2594.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2630.1     | 1948.216 | 1.948  | 1846.3                | 2653.6                   |

Final op time: 2.546 us (3437.3 cycles), bucket=NCRISC gather

