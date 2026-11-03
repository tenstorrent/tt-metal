# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_33_13
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

Report: generated/profiler/reports/2026_11_02_09_33_31/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3431.0     | 2541.457 | 2.541  | 1999.0     | 3424.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1811.8     | 1342.074 | 1.342  | 1751.0     | 1792.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3368.9     | 2495.481 | 2.495  | 2036.0     | 3371.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3454.1     | 2558.617 | 2.559  | 2093.0     | 3448.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3386.6     | 2508.568 | 2.509  | 1997.0     | 3366.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1842.6     | 1364.889 | 1.365  | 1794.0     | 1844.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3415.1     | 2529.679 | 2.530  | 2052.0     | 3490.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3305.7     | 2448.691 | 2.449  | 2024.0     | 3325.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1942.0     | 1438.494 | 1.438  | 1829.0     | 1946.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3395.0     | 2514.815 | 2.515  | 2186.0     | 3432.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1874.3     | 1388.370 | 1.388  | 1826.0     | 1875.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3350.1     | 2481.556 | 2.482  | 1976.0     | 3369.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3423.8     | 2536.142 | 2.536  | 3395.0                | 3423.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2594.6     | 1921.926 | 1.922  | 1811.8                | 2590.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2625.9     | 1945.105 | 1.945  | 1842.6                | 2646.0                   |

Final op time: 2.536 us (3423.8 cycles), bucket=NCRISC gather

