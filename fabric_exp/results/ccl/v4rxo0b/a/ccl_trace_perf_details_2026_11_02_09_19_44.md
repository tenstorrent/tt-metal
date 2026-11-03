# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_19_44
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

Report: generated/profiler/reports/2026_11_02_09_20_01/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3405.9     | 2522.864 | 2.523  | 2056.0     | 3378.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1798.8     | 1332.469 | 1.332  | 1760.0     | 1791.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3343.0     | 2476.272 | 2.476  | 1998.0     | 3317.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3430.9     | 2541.432 | 2.541  | 2150.0     | 3351.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3359.3     | 2488.346 | 2.488  | 1995.0     | 3320.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1843.1     | 1365.235 | 1.365  | 1780.0     | 1840.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3408.9     | 2525.086 | 2.525  | 2039.0     | 3773.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3300.9     | 2445.111 | 2.445  | 2033.0     | 3271.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1911.1     | 1415.654 | 1.416  | 1807.0     | 1910.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3387.2     | 2509.037 | 2.509  | 2130.0     | 3348.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1883.9     | 1395.481 | 1.395  | 1809.0     | 1881.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3347.0     | 2479.284 | 2.479  | 1996.0     | 3352.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3408.2     | 2524.605 | 2.525  | 3387.2                | 3407.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2585.7     | 1915.352 | 1.915  | 1798.8                | 2592.4                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2611.1     | 1934.111 | 1.934  | 1843.1                | 2627.1                   |

Final op time: 2.525 us (3408.2 cycles), bucket=NCRISC gather

