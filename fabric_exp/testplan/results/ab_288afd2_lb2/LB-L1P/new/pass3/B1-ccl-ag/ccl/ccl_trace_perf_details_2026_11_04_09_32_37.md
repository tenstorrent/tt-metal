# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_32_37
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

Report: generated/profiler/reports/2026_11_04_09_32_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3389.3     | 2510.617 | 2.511  | 2002.0     | 3338.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1934.9     | 1433.284 | 1.433  | 1866.0     | 1925.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3345.0     | 2477.778 | 2.478  | 1999.0     | 3371.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3417.0     | 2531.086 | 2.531  | 2035.0     | 3432.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3328.3     | 2465.407 | 2.465  | 1945.0     | 3292.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1862.0     | 1379.235 | 1.379  | 1829.0     | 1856.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3398.6     | 2517.481 | 2.517  | 1970.0     | 3705.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3309.2     | 2451.284 | 2.451  | 2084.0     | 3330.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1876.0     | 1389.654 | 1.390  | 1824.0     | 1870.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3377.6     | 2501.926 | 2.502  | 2132.0     | 3350.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1917.0     | 1420.000 | 1.420  | 1861.0     | 1918.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3331.4     | 2467.704 | 2.468  | 2043.0     | 3359.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3395.6     | 2515.278 | 2.515  | 3377.6                | 3394.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2622.4     | 1942.494 | 1.942  | 1917.0                | 2622.1                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2603.6     | 1928.593 | 1.929  | 1862.0                | 2603.7                   |

Final op time: 2.515 us (3395.6 cycles), bucket=NCRISC gather

