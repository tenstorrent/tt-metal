# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_39
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

Report: generated/profiler/reports/2026_11_04_09_27_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3496.9     | 2590.296 | 2.590  | 2217.0     | 3530.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1963.3     | 1454.296 | 1.454  | 1867.0     | 1949.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3448.8     | 2554.691 | 2.555  | 2486.0     | 3470.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3530.5     | 2615.185 | 2.615  | 2402.0     | 3481.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3431.8     | 2542.049 | 2.542  | 2098.0     | 3352.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1906.7     | 1412.395 | 1.412  | 1783.0     | 1884.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3479.2     | 2577.185 | 2.577  | 2221.0     | 3465.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3397.3     | 2516.494 | 2.516  | 2468.0     | 3400.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1876.5     | 1389.975 | 1.390  | 1784.0     | 1871.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3472.4     | 2572.148 | 2.572  | 2379.0     | 3558.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1930.5     | 1430.000 | 1.430  | 1824.0     | 1925.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3414.7     | 2529.432 | 2.529  | 2031.0     | 3371.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3494.8     | 2588.704 | 2.589  | 3472.4                | 3488.1                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2680.7     | 1985.710 | 1.986  | 1930.5                | 2680.3                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2661.7     | 1971.623 | 1.972  | 1876.5                | 2660.7                   |

Final op time: 2.589 us (3494.8 cycles), bucket=NCRISC gather

