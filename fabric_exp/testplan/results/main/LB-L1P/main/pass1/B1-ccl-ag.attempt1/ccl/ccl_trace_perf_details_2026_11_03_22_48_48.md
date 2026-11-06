# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_48_48
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

Report: generated/profiler/reports/2026_11_03_22_49_02/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3584.2     | 2654.963 | 2.655  | 2149.0     | 3566.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1902.4     | 1409.185 | 1.409  | 1777.0     | 1876.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3509.4     | 2599.556 | 2.600  | 2192.0     | 3507.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3605.2     | 2670.543 | 2.671  | 2241.0     | 3520.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3529.2     | 2614.247 | 2.614  | 2162.0     | 3421.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1897.6     | 1405.654 | 1.406  | 1803.0     | 1900.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3550.1     | 2629.679 | 2.630  | 2062.0     | 3627.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3446.2     | 2552.716 | 2.553  | 2016.0     | 3420.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1999.7     | 1481.259 | 1.481  | 1834.0     | 1976.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3511.2     | 2600.889 | 2.601  | 1725.0     | 3526.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1906.2     | 1411.975 | 1.412  | 1831.0     | 1901.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3468.3     | 2569.086 | 2.569  | 2006.0     | 3458.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3562.7     | 2639.019 | 2.639  | 3511.2                | 3567.1                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2696.0     | 1997.031 | 1.997  | 1902.4                | 2676.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2718.8     | 2013.889 | 2.014  | 1897.6                | 2734.0                   |

Final op time: 2.639 us (3562.7 cycles), bucket=NCRISC gather

