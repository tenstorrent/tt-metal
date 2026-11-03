# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_08_38_22
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

Report: generated/profiler/reports/2026_11_02_08_38_37/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3531.7     | 2616.099 | 2.616  | 2103.0     | 3465.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1853.2     | 1372.741 | 1.373  | 1767.0     | 1819.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3462.5     | 2564.790 | 2.565  | 2161.0     | 3255.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3554.1     | 2632.691 | 2.633  | 2214.0     | 3702.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3479.3     | 2577.259 | 2.577  | 2070.0     | 3409.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1866.8     | 1382.815 | 1.383  | 1812.0     | 1871.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3547.6     | 2627.877 | 2.628  | 2089.0     | 3845.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3442.9     | 2550.296 | 2.550  | 1987.0     | 3540.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2013.5     | 1491.481 | 1.491  | 1800.0     | 2004.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3529.3     | 2614.321 | 2.614  | 2240.0     | 3475.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1905.8     | 1411.728 | 1.412  | 1837.0     | 1896.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3475.0     | 2574.099 | 2.574  | 2064.0     | 3433.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3540.7     | 2622.747 | 2.623  | 3529.3                | 3539.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2670.3     | 1978.006 | 1.978  | 1853.2                | 2674.4                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2704.4     | 2003.296 | 2.003  | 1866.8                | 2738.0                   |

Final op time: 2.623 us (3540.7 cycles), bucket=NCRISC gather

