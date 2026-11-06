# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_55
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

Report: generated/profiler/reports/2026_11_04_09_28_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3507.9     | 2598.420 | 2.598  | 2210.0     | 3562.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1957.9     | 1450.321 | 1.450  | 1845.0     | 1958.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3466.1     | 2567.506 | 2.568  | 2101.0     | 3373.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3545.3     | 2626.123 | 2.626  | 2143.0     | 3471.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3447.8     | 2553.926 | 2.554  | 2040.0     | 3449.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1925.1     | 1425.975 | 1.426  | 1793.0     | 1937.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3484.5     | 2581.086 | 2.581  | 2138.0     | 3487.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3397.6     | 2516.741 | 2.517  | 2089.0     | 3393.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1918.3     | 1420.963 | 1.421  | 1800.0     | 1885.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3468.2     | 2569.062 | 2.569  | 2226.0     | 3561.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1923.8     | 1425.012 | 1.425  | 1850.0     | 1908.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3413.0     | 2528.123 | 2.528  | 2126.0     | 3453.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3501.5     | 2593.673 | 2.594  | 3468.2                | 3496.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2681.8     | 1986.500 | 1.987  | 1923.8                | 2677.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2680.6     | 1985.642 | 1.986  | 1918.3                | 2669.0                   |

Final op time: 2.594 us (3501.5 cycles), bucket=NCRISC gather

