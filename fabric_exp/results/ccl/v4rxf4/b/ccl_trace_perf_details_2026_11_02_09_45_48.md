# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_45_48
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

Report: generated/profiler/reports/2026_11_02_09_46_00/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3390.1     | 2511.210 | 2.511  | 2063.0     | 3387.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1802.8     | 1335.383 | 1.335  | 1747.0     | 1782.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3330.2     | 2466.815 | 2.467  | 2095.0     | 3310.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3413.3     | 2528.395 | 2.528  | 2125.0     | 3483.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3343.7     | 2476.790 | 2.477  | 2003.0     | 3319.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1838.9     | 1362.173 | 1.362  | 1797.0     | 1842.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3402.1     | 2520.074 | 2.520  | 1945.0     | 3663.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3290.1     | 2437.086 | 2.437  | 2027.0     | 3343.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1909.2     | 1414.247 | 1.414  | 1812.0     | 1867.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3385.8     | 2508.025 | 2.508  | 2140.0     | 3437.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1875.0     | 1388.864 | 1.389  | 1835.0     | 1877.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3345.7     | 2478.321 | 2.478  | 2044.0     | 3362.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3397.9     | 2516.926 | 2.517  | 3385.8                | 3396.1                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2577.9     | 1909.531 | 1.910  | 1802.8                | 2582.5                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2606.0     | 1930.389 | 1.930  | 1838.9                | 2619.7                   |

Final op time: 2.517 us (3397.9 cycles), bucket=NCRISC gather

