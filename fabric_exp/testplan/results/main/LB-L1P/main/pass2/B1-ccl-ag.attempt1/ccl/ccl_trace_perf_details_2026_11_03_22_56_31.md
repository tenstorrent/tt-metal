# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_56_31
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

Report: generated/profiler/reports/2026_11_03_22_56_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3541.1     | 2623.037 | 2.623  | 2130.0     | 3573.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1894.1     | 1403.037 | 1.403  | 1797.0     | 1851.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3478.4     | 2576.593 | 2.577  | 2176.0     | 3484.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3591.6     | 2660.420 | 2.660  | 2242.0     | 3557.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3511.7     | 2601.259 | 2.601  | 2038.0     | 3499.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1897.4     | 1405.506 | 1.406  | 1793.0     | 1891.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3549.3     | 2629.136 | 2.629  | 2135.0     | 3809.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3438.7     | 2547.160 | 2.547  | 2029.0     | 3452.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1984.1     | 1469.679 | 1.470  | 1810.0     | 1957.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3522.9     | 2609.556 | 2.610  | 2141.0     | 3479.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1940.7     | 1437.556 | 1.438  | 1853.0     | 1946.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3484.4     | 2581.062 | 2.581  | 2006.0     | 3543.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3551.2     | 2630.537 | 2.631  | 3522.9                | 3545.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2696.3     | 1997.253 | 1.997  | 1894.1                | 2689.7                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2711.1     | 2008.210 | 2.008  | 1897.4                | 2731.2                   |

Final op time: 2.631 us (3551.2 cycles), bucket=NCRISC gather

