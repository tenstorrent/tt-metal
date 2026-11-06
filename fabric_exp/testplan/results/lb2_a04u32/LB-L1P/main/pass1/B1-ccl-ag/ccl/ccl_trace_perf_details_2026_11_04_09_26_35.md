# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_26_35
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

Report: generated/profiler/reports/2026_11_04_09_26_48/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3461.4     | 2564.000 | 2.564  | 2124.0     | 3418.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1858.9     | 1376.988 | 1.377  | 1776.0     | 1845.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3391.7     | 2512.370 | 2.512  | 2129.0     | 3480.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3476.1     | 2574.864 | 2.575  | 2057.0     | 3371.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3392.9     | 2513.284 | 2.513  | 2034.0     | 3376.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1891.5     | 1401.136 | 1.401  | 1805.0     | 1889.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3472.2     | 2572.000 | 2.572  | 2127.0     | 3703.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3365.9     | 2493.235 | 2.493  | 2161.0     | 3417.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1979.4     | 1466.222 | 1.466  | 1799.0     | 1941.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3438.2     | 2546.790 | 2.547  | 2194.0     | 3373.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1791.8     | 1327.284 | 1.327  | 1751.0     | 1786.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3403.1     | 2520.840 | 2.521  | 2049.0     | 3351.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3462.0     | 2564.414 | 2.564  | 3438.2                | 3466.8                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2602.4     | 1927.698 | 1.928  | 1791.8                | 2612.4                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2666.4     | 1975.142 | 1.975  | 1891.5                | 2685.6                   |

Final op time: 2.564 us (3462.0 cycles), bucket=NCRISC gather

