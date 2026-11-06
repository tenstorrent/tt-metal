# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_26_20
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

Report: generated/profiler/reports/2026_11_04_09_26_34/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3462.5     | 2564.790 | 2.565  | 2110.0     | 3465.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1836.6     | 1360.444 | 1.360  | 1786.0     | 1825.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3397.1     | 2516.395 | 2.516  | 2069.0     | 3344.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3474.7     | 2573.877 | 2.574  | 2137.0     | 3395.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3393.2     | 2513.481 | 2.513  | 2010.0     | 3321.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1901.8     | 1408.716 | 1.409  | 1796.0     | 1905.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3443.7     | 2550.864 | 2.551  | 2078.0     | 3559.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3336.9     | 2471.802 | 2.472  | 1989.0     | 3362.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1985.4     | 1470.642 | 1.471  | 1805.0     | 1973.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3416.2     | 2530.519 | 2.531  | 2085.0     | 3428.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1806.3     | 1337.975 | 1.338  | 1749.0     | 1792.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3372.7     | 2498.296 | 2.498  | 2062.0     | 3410.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3449.3     | 2555.012 | 2.555  | 3416.2                | 3453.1                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2593.2     | 1920.926 | 1.921  | 1806.3                | 2586.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2664.2     | 1973.512 | 1.974  | 1901.8                | 2679.0                   |

Final op time: 2.555 us (3449.3 cycles), bucket=NCRISC gather

