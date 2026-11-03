# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_45_22
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_45_34/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3879.7     | 2873.852 | 2.874  | 2634.0     | 3844.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2195.1     | 1625.975 | 1.626  | 2110.0     | 2177.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3809.6     | 2821.951 | 2.822  | 2590.0     | 3825.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3901.1     | 2889.704 | 2.890  | 2692.0     | 3832.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3845.8     | 2848.765 | 2.849  | 2596.0     | 3812.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2175.8     | 1611.728 | 1.612  | 2128.0     | 2174.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3856.0     | 2856.296 | 2.856  | 2556.0     | 3950.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3740.9     | 2771.012 | 2.771  | 2555.0     | 3764.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2266.1     | 1678.568 | 1.679  | 2138.0     | 2246.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3823.9     | 2832.494 | 2.832  | 2451.0     | 3886.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2238.4     | 1658.074 | 1.658  | 2171.0     | 2236.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3789.4     | 2806.988 | 2.807  | 2546.0     | 3803.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3865.2     | 2863.086 | 2.863  | 3823.9                | 3867.8                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3005.0     | 2225.957 | 2.226  | 2195.1                | 2989.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3010.2     | 2229.809 | 2.230  | 2175.8                | 3027.8                   |

Final op time: 2.863 us (3865.2 cycles), bucket=NCRISC gather

