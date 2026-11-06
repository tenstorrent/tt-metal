# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_31_23
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

Report: generated/profiler/reports/2026_11_04_09_31_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3345.4     | 2478.074 | 2.478  | 2014.0     | 3348.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1820.4     | 1348.420 | 1.348  | 1780.0     | 1809.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3278.2     | 2428.296 | 2.428  | 2088.0     | 3274.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3367.1     | 2494.123 | 2.494  | 2143.0     | 3333.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3287.7     | 2435.333 | 2.435  | 2068.0     | 3252.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1858.0     | 1376.272 | 1.376  | 1824.0     | 1854.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3348.2     | 2480.123 | 2.480  | 2120.0     | 3591.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3249.7     | 2407.160 | 2.407  | 2051.0     | 3251.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1903.2     | 1409.753 | 1.410  | 1836.0     | 1887.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3325.1     | 2463.062 | 2.463  | 2105.0     | 3331.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1782.0     | 1320.000 | 1.320  | 1749.0     | 1778.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3290.9     | 2437.704 | 2.438  | 2038.0     | 3291.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3346.4     | 2478.846 | 2.479  | 3325.1                | 3346.8                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2534.9     | 1877.728 | 1.878  | 1782.0                | 2535.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2582.6     | 1913.006 | 1.913  | 1858.0                | 2590.7                   |

Final op time: 2.479 us (3346.4 cycles), bucket=NCRISC gather

