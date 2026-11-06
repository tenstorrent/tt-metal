# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_31_40
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

Report: generated/profiler/reports/2026_11_04_09_31_56/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3348.2     | 2480.148 | 2.480  | 2057.0     | 3320.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1809.7     | 1340.494 | 1.340  | 1768.0     | 1811.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3284.6     | 2433.062 | 2.433  | 2048.0     | 3288.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3373.1     | 2498.617 | 2.499  | 2036.0     | 3425.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3297.0     | 2442.198 | 2.442  | 2031.0     | 3256.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1866.8     | 1382.815 | 1.383  | 1812.0     | 1864.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3354.4     | 2484.741 | 2.485  | 2073.0     | 3670.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3251.7     | 2408.691 | 2.409  | 2035.0     | 3297.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1924.7     | 1425.728 | 1.426  | 1832.0     | 1895.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3322.7     | 2461.235 | 2.461  | 2072.0     | 3342.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1783.4     | 1321.062 | 1.321  | 1749.0     | 1784.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3296.8     | 2442.049 | 2.442  | 2050.0     | 3301.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3349.6     | 2481.185 | 2.481  | 3322.7                | 3351.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2535.5     | 1878.111 | 1.878  | 1783.4                | 2530.7                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2593.2     | 1920.914 | 1.921  | 1866.8                | 2604.7                   |

Final op time: 2.481 us (3349.6 cycles), bucket=NCRISC gather

