# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_48_34
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

Report: generated/profiler/reports/2026_11_03_22_48_48/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3580.3     | 2652.074 | 2.652  | 2119.0     | 3614.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1881.8     | 1393.951 | 1.394  | 1768.0     | 1848.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3490.6     | 2585.654 | 2.586  | 2071.0     | 3413.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3601.8     | 2668.000 | 2.668  | 2248.0     | 3770.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3524.9     | 2611.012 | 2.611  | 2026.0     | 3479.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1883.8     | 1395.432 | 1.395  | 1785.0     | 1883.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3568.7     | 2643.506 | 2.644  | 2123.0     | 3797.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3482.0     | 2579.284 | 2.579  | 2040.0     | 3490.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1998.5     | 1480.370 | 1.480  | 1813.0     | 2003.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3542.6     | 2624.173 | 2.624  | 2149.0     | 3508.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1926.5     | 1427.012 | 1.427  | 1825.0     | 1937.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3493.3     | 2587.654 | 2.588  | 2016.0     | 3434.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3573.4     | 2646.938 | 2.647  | 3542.6                | 3574.5                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2703.8     | 2002.815 | 2.003  | 1881.8                | 2704.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2716.6     | 2012.278 | 2.012  | 1883.8                | 2744.6                   |

Final op time: 2.647 us (3573.4 cycles), bucket=NCRISC gather

