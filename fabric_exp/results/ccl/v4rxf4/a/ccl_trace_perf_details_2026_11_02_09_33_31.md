# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_33_31
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

Report: generated/profiler/reports/2026_11_02_09_33_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3412.6     | 2527.827 | 2.528  | 2090.0     | 3438.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1805.9     | 1337.679 | 1.338  | 1746.0     | 1781.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3344.0     | 2477.037 | 2.477  | 2113.0     | 3369.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3437.6     | 2546.370 | 2.546  | 2115.0     | 3410.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3374.7     | 2499.753 | 2.500  | 2038.0     | 3349.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1847.8     | 1368.765 | 1.369  | 1796.0     | 1853.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3405.9     | 2522.864 | 2.523  | 2096.0     | 3596.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3289.3     | 2436.519 | 2.437  | 2011.0     | 3296.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1916.3     | 1419.457 | 1.419  | 1812.0     | 1896.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3379.4     | 2503.259 | 2.503  | 2123.0     | 3400.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1874.9     | 1388.790 | 1.389  | 1827.0     | 1872.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3351.0     | 2482.198 | 2.482  | 2033.0     | 3390.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3408.9     | 2525.080 | 2.525  | 3379.4                | 3409.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2586.2     | 1915.685 | 1.916  | 1805.9                | 2582.1                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2614.8     | 1936.864 | 1.937  | 1847.8                | 2630.1                   |

Final op time: 2.525 us (3408.9 cycles), bucket=NCRISC gather

