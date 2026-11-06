# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_32_56
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

Report: generated/profiler/reports/2026_11_04_09_33_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3404.2     | 2521.654 | 2.522  | 2031.0     | 3386.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1936.8     | 1434.667 | 1.435  | 1874.0     | 1920.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3359.6     | 2488.617 | 2.489  | 2156.0     | 3355.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3441.2     | 2549.012 | 2.549  | 2115.0     | 3396.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3350.5     | 2481.827 | 2.482  | 1971.0     | 3299.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1872.0     | 1386.642 | 1.387  | 1813.0     | 1870.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3398.6     | 2517.457 | 2.517  | 2124.0     | 3513.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3306.8     | 2449.506 | 2.450  | 2179.0     | 3272.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1868.3     | 1383.926 | 1.384  | 1815.0     | 1865.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3372.6     | 2498.198 | 2.498  | 2249.0     | 3349.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1902.5     | 1409.259 | 1.409  | 1845.0     | 1898.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3321.3     | 2460.222 | 2.460  | 2006.0     | 3338.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3404.1     | 2521.580 | 2.522  | 3372.6                | 3401.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2624.2     | 1943.815 | 1.944  | 1902.5                | 2621.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2605.3     | 1929.852 | 1.930  | 1868.3                | 2596.6                   |

Final op time: 2.522 us (3404.1 cycles), bucket=NCRISC gather

