# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_33_48
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

Report: generated/profiler/reports/2026_11_02_09_34_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3427.2     | 2538.691 | 2.539  | 2070.0     | 3488.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1816.4     | 1345.506 | 1.346  | 1748.0     | 1796.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3361.3     | 2489.877 | 2.490  | 2151.0     | 3363.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3452.9     | 2557.728 | 2.558  | 2197.0     | 3384.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3387.1     | 2508.988 | 2.509  | 1994.0     | 3355.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1837.9     | 1361.432 | 1.361  | 1798.0     | 1838.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3407.6     | 2524.123 | 2.524  | 2070.0     | 3543.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3290.6     | 2437.506 | 2.438  | 1978.0     | 3288.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1946.8     | 1442.074 | 1.442  | 1814.0     | 1928.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3378.2     | 2502.346 | 2.502  | 1999.0     | 3389.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1868.3     | 1383.926 | 1.384  | 1814.0     | 1867.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3349.0     | 2480.741 | 2.481  | 2025.0     | 3389.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3416.5     | 2530.722 | 2.531  | 3378.2                | 3417.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2590.6     | 1918.981 | 1.919  | 1816.4                | 2579.5                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2623.8     | 1943.531 | 1.944  | 1837.9                | 2647.9                   |

Final op time: 2.531 us (3416.5 cycles), bucket=NCRISC gather

