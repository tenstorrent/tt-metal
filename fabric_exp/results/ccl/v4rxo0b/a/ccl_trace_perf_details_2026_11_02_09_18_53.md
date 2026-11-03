# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_18_53
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

Report: generated/profiler/reports/2026_11_02_09_19_10/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3428.8     | 2539.877 | 2.540  | 2114.0     | 3376.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1815.1     | 1344.543 | 1.345  | 1751.0     | 1790.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3342.7     | 2476.049 | 2.476  | 2073.0     | 3401.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3458.0     | 2561.506 | 2.562  | 2100.0     | 3813.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3389.0     | 2510.346 | 2.510  | 2042.0     | 3304.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1848.9     | 1369.556 | 1.370  | 1789.0     | 1854.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3437.2     | 2546.099 | 2.546  | 2084.0     | 4082.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3321.2     | 2460.123 | 2.460  | 1999.0     | 3314.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1940.1     | 1437.111 | 1.437  | 1819.0     | 1903.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3391.9     | 2512.543 | 2.513  | 1763.0     | 3378.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1877.2     | 1390.494 | 1.390  | 1838.0     | 1876.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3355.5     | 2485.556 | 2.486  | 2063.0     | 3356.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3429.0     | 2540.006 | 2.540  | 3391.9                | 3433.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2600.6     | 1926.377 | 1.926  | 1815.1                | 2599.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2621.8     | 1942.068 | 1.942  | 1848.9                | 2641.4                   |

Final op time: 2.540 us (3429.0 cycles), bucket=NCRISC gather

