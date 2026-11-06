# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_56_45
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

Report: generated/profiler/reports/2026_11_03_22_56_59/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3547.1     | 2627.481 | 2.627  | 2197.0     | 3522.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1866.4     | 1382.494 | 1.382  | 1774.0     | 1822.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3468.1     | 2568.963 | 2.569  | 2150.0     | 3499.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3593.8     | 2662.099 | 2.662  | 2059.0     | 3569.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3522.7     | 2609.432 | 2.609  | 1988.0     | 3460.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1875.6     | 1389.309 | 1.389  | 1789.0     | 1884.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3557.4     | 2635.136 | 2.635  | 2088.0     | 3837.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3444.4     | 2551.407 | 2.551  | 2144.0     | 3483.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1984.3     | 1469.877 | 1.470  | 1811.0     | 1961.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3514.4     | 2603.284 | 2.603  | 2138.0     | 3478.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1934.6     | 1433.012 | 1.433  | 1825.0     | 1922.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3488.2     | 2583.877 | 2.584  | 2123.0     | 3524.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3553.2     | 2632.000 | 2.632  | 3514.4                | 3552.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2692.0     | 1994.086 | 1.994  | 1866.4                | 2689.5                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2704.1     | 2003.006 | 2.003  | 1875.6                | 2726.2                   |

Final op time: 2.632 us (3553.2 cycles), bucket=NCRISC gather

