# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_52_35
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

Report: generated/profiler/reports/2026_11_02_09_52_48/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3441.5     | 2549.259 | 2.549  | 2304.0     | 3368.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1819.7     | 1347.951 | 1.348  | 1743.0     | 1787.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3359.9     | 2488.815 | 2.489  | 2370.0     | 3422.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3468.1     | 2568.963 | 2.569  | 2457.0     | 3669.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3395.8     | 2515.407 | 2.515  | 2229.0     | 3317.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1834.2     | 1358.691 | 1.359  | 1778.0     | 1833.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3431.9     | 2542.123 | 2.542  | 2370.0     | 3784.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3312.2     | 2453.506 | 2.454  | 2484.0     | 3285.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1955.9     | 1448.815 | 1.449  | 1794.0     | 1960.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3393.8     | 2513.951 | 2.514  | 2348.0     | 3373.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1866.6     | 1382.691 | 1.383  | 1798.0     | 1861.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3358.5     | 2487.778 | 2.488  | 2230.0     | 3361.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3433.8     | 2543.574 | 2.544  | 3393.8                | 3436.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2598.6     | 1924.889 | 1.925  | 1819.7                | 2589.4                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2627.1     | 1946.025 | 1.946  | 1834.2                | 2657.2                   |

Final op time: 2.544 us (3433.8 cycles), bucket=NCRISC gather

