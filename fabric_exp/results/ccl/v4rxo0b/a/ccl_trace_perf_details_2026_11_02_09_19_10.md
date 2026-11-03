# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_19_10
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

Report: generated/profiler/reports/2026_11_02_09_19_27/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3400.9     | 2519.160 | 2.519  | 2039.0     | 3377.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1802.2     | 1334.988 | 1.335  | 1735.0     | 1778.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3341.1     | 2474.914 | 2.475  | 2361.0     | 3341.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3434.8     | 2544.272 | 2.544  | 2380.0     | 3365.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3359.6     | 2488.593 | 2.489  | 1993.0     | 3310.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1839.9     | 1362.914 | 1.363  | 1798.0     | 1837.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3409.0     | 2525.210 | 2.525  | 2334.0     | 3410.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3287.2     | 2434.938 | 2.435  | 2373.0     | 3340.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1918.0     | 1420.741 | 1.421  | 1821.0     | 1907.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3379.4     | 2503.284 | 2.503  | 2506.0     | 3410.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1876.7     | 1390.148 | 1.390  | 1830.0     | 1870.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3343.2     | 2476.444 | 2.476  | 2014.0     | 3349.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3406.0     | 2522.981 | 2.523  | 3379.4                | 3404.9                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2581.4     | 1912.167 | 1.912  | 1802.2                | 2581.9                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2610.6     | 1933.753 | 1.934  | 1839.9                | 2629.6                   |

Final op time: 2.523 us (3406.0 cycles), bucket=NCRISC gather

