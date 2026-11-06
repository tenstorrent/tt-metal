# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_31_58
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

Report: generated/profiler/reports/2026_11_04_09_32_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3401.8     | 2519.877 | 2.520  | 2082.0     | 3340.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1948.4     | 1443.284 | 1.443  | 1870.0     | 1938.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3361.2     | 2489.802 | 2.490  | 2131.0     | 3474.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3494.2     | 2588.272 | 2.588  | 2149.0     | 3534.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3402.7     | 2520.494 | 2.520  | 2008.0     | 3257.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1861.1     | 1378.568 | 1.379  | 1825.0     | 1865.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3394.8     | 2514.691 | 2.515  | 2157.0     | 3740.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3307.5     | 2450.000 | 2.450  | 2028.0     | 3279.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1877.0     | 1390.370 | 1.390  | 1824.0     | 1861.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3370.7     | 2496.790 | 2.497  | 1926.0     | 3363.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1906.1     | 1411.901 | 1.412  | 1860.0     | 1903.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3321.0     | 2459.975 | 2.460  | 2064.0     | 3322.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3415.4     | 2529.907 | 2.530  | 3370.7                | 3398.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2641.2     | 1956.420 | 1.956  | 1906.1                | 2628.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2605.1     | 1929.679 | 1.930  | 1861.1                | 2599.0                   |

Final op time: 2.530 us (3415.4 cycles), bucket=NCRISC gather

