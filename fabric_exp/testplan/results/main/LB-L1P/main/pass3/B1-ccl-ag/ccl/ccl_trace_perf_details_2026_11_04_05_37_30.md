# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_37_30
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

Report: generated/profiler/reports/2026_11_04_05_37_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3575.7     | 2648.691 | 2.649  | 2105.0     | 3475.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1912.1     | 1416.370 | 1.416  | 1782.0     | 1853.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3498.2     | 2591.284 | 2.591  | 2209.0     | 3433.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3614.9     | 2677.704 | 2.678  | 2135.0     | 3707.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3538.5     | 2621.086 | 2.621  | 2034.0     | 3415.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1889.5     | 1399.654 | 1.400  | 1800.0     | 1892.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3551.3     | 2630.568 | 2.631  | 2136.0     | 3977.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3442.3     | 2549.827 | 2.550  | 2107.0     | 3347.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1984.2     | 1469.802 | 1.470  | 1811.0     | 1974.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3503.8     | 2595.383 | 2.595  | 2196.0     | 3477.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1934.4     | 1432.864 | 1.433  | 1838.0     | 1931.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3465.3     | 2566.889 | 2.567  | 2075.0     | 3444.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3561.4     | 2638.086 | 2.638  | 3503.8                | 3563.5                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2706.8     | 2005.037 | 2.005  | 1912.1                | 2688.3                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2709.3     | 2006.907 | 2.007  | 1889.5                | 2724.8                   |

Final op time: 2.638 us (3561.4 cycles), bucket=NCRISC gather

