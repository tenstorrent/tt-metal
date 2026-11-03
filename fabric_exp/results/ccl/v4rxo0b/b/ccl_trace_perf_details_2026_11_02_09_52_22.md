# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_52_22
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

Report: generated/profiler/reports/2026_11_02_09_52_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3413.4     | 2528.444 | 2.528  | 2063.0     | 3378.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1811.9     | 1342.173 | 1.342  | 1744.0     | 1786.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3355.2     | 2485.358 | 2.485  | 2131.0     | 3429.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3444.5     | 2551.457 | 2.551  | 2186.0     | 3418.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3376.1     | 2500.815 | 2.501  | 1999.0     | 3341.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1843.5     | 1365.531 | 1.366  | 1803.0     | 1840.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3408.5     | 2524.840 | 2.525  | 2105.0     | 3705.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3299.9     | 2444.395 | 2.444  | 2072.0     | 3292.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1957.9     | 1450.321 | 1.450  | 1798.0     | 1967.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3390.5     | 2511.481 | 2.511  | 2186.0     | 3416.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1875.3     | 1389.111 | 1.389  | 1821.0     | 1871.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3355.9     | 2485.877 | 2.486  | 2031.0     | 3369.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3414.2     | 2529.056 | 2.529  | 3390.5                | 3411.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2590.8     | 1919.123 | 1.919  | 1811.9                | 2587.6                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2628.1     | 1946.772 | 1.947  | 1843.5                | 2656.6                   |

Final op time: 2.529 us (3414.2 cycles), bucket=NCRISC gather

