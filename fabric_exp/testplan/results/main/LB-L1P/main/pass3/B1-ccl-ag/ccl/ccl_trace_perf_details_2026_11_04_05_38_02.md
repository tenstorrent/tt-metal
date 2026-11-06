# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_38_02
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

Report: generated/profiler/reports/2026_11_04_05_38_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3543.6     | 2624.914 | 2.625  | 2182.0     | 3555.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1878.2     | 1391.284 | 1.391  | 1787.0     | 1846.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3475.4     | 2574.346 | 2.574  | 2163.0     | 3464.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3587.4     | 2657.333 | 2.657  | 2241.0     | 3481.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3500.7     | 2593.111 | 2.593  | 2159.0     | 3459.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1891.4     | 1401.037 | 1.401  | 1805.0     | 1882.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3535.4     | 2618.815 | 2.619  | 2220.0     | 3641.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3428.1     | 2539.333 | 2.539  | 2053.0     | 3413.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2018.5     | 1495.185 | 1.495  | 1813.0     | 2008.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3505.5     | 2596.642 | 2.597  | 2141.0     | 3465.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1912.1     | 1416.370 | 1.416  | 1840.0     | 1905.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3475.5     | 2574.444 | 2.574  | 2099.0     | 3464.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3543.0     | 2624.426 | 2.624  | 3505.5                | 3539.5                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2679.8     | 1985.025 | 1.985  | 1878.2                | 2670.1                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2715.2     | 2011.253 | 2.011  | 1891.4                | 2746.9                   |

Final op time: 2.624 us (3543.0 cycles), bucket=NCRISC gather

