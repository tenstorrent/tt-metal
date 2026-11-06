# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_03
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

Report: generated/profiler/reports/2026_11_04_09_27_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3481.9     | 2579.185 | 2.579  | 2097.0     | 3460.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1856.7     | 1375.309 | 1.375  | 1786.0     | 1835.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3413.1     | 2528.222 | 2.528  | 2074.0     | 3503.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3508.0     | 2598.494 | 2.598  | 2008.0     | 3596.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3424.9     | 2536.963 | 2.537  | 1992.0     | 3417.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1891.8     | 1401.309 | 1.401  | 1783.0     | 1896.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3462.9     | 2565.086 | 2.565  | 2070.0     | 3747.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3358.6     | 2487.827 | 2.488  | 1994.0     | 3238.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2008.2     | 1487.531 | 1.488  | 1820.0     | 2003.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3427.6     | 2538.938 | 2.539  | 1811.0     | 3372.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1799.6     | 1333.012 | 1.333  | 1749.0     | 1794.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3377.2     | 2501.605 | 2.502  | 2040.0     | 3350.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3470.1     | 2570.426 | 2.570  | 3427.6                | 3472.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2609.9     | 1933.278 | 1.933  | 1799.6                | 2607.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2672.5     | 1979.667 | 1.980  | 1891.8                | 2692.7                   |

Final op time: 2.570 us (3470.1 cycles), bucket=NCRISC gather

