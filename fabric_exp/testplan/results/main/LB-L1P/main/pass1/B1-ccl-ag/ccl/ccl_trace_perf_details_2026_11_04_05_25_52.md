# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_25_52
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

Report: generated/profiler/reports/2026_11_04_05_26_10/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3666.0     | 2715.531 | 2.716  | 2128.0     | 3554.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1877.6     | 1390.790 | 1.391  | 1783.0     | 1840.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3545.5     | 2626.296 | 2.626  | 2138.0     | 3417.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3745.0     | 2774.074 | 2.774  | 2218.0     | 3685.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3672.4     | 2720.321 | 2.720  | 2109.0     | 3487.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1879.3     | 1392.099 | 1.392  | 1801.0     | 1868.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3592.8     | 2661.333 | 2.661  | 2236.0     | 4011.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3434.0     | 2543.679 | 2.544  | 2043.0     | 3313.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1980.0     | 1466.691 | 1.467  | 1797.0     | 1934.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3496.3     | 2589.877 | 2.590  | 665.0      | 3462.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1923.4     | 1424.716 | 1.425  | 1817.0     | 1918.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3469.3     | 2569.827 | 2.570  | 2156.0     | 3440.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3625.0     | 2685.204 | 2.685  | 3496.3                | 3629.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2726.8     | 2019.877 | 2.020  | 1877.6                | 2678.7                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2718.5     | 2013.728 | 2.014  | 1879.3                | 2724.7                   |

Final op time: 2.685 us (3625.0 cycles), bucket=NCRISC gather

