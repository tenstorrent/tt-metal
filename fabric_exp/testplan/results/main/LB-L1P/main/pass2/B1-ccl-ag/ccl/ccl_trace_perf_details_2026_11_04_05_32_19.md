# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_32_19
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

Report: generated/profiler/reports/2026_11_04_05_32_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3525.9     | 2611.802 | 2.612  | 2096.0     | 3585.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1866.6     | 1382.667 | 1.383  | 1770.0     | 1855.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3454.2     | 2558.691 | 2.559  | 2157.0     | 3541.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3545.4     | 2626.222 | 2.626  | 2120.0     | 3615.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3470.4     | 2570.691 | 2.571  | 2127.0     | 3474.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1888.9     | 1399.210 | 1.399  | 1803.0     | 1876.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3519.0     | 2606.691 | 2.607  | 2197.0     | 3759.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3421.8     | 2534.667 | 2.535  | 2131.0     | 3449.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1984.2     | 1469.753 | 1.470  | 1802.0     | 1945.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3495.2     | 2589.037 | 2.589  | 2233.0     | 3511.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1920.5     | 1422.593 | 1.423  | 1845.0     | 1927.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3462.6     | 2564.889 | 2.565  | 2081.0     | 3500.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3521.4     | 2608.438 | 2.608  | 3495.2                | 3522.5                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2669.8     | 1977.654 | 1.978  | 1866.6                | 2671.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2697.5     | 1998.136 | 1.998  | 1888.9                | 2719.2                   |

Final op time: 2.608 us (3521.4 cycles), bucket=NCRISC gather

