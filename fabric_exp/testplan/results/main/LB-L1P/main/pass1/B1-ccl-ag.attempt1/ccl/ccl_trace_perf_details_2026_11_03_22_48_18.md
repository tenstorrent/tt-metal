# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_48_18
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

Report: generated/profiler/reports/2026_11_03_22_48_33/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3552.7     | 2631.630 | 2.632  | 2095.0     | 3554.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1892.5     | 1401.877 | 1.402  | 1777.0     | 1863.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3476.4     | 2575.086 | 2.575  | 2019.0     | 3411.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3577.0     | 2649.654 | 2.650  | 2263.0     | 3556.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3494.8     | 2588.716 | 2.589  | 2023.0     | 3484.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1887.1     | 1397.827 | 1.398  | 1786.0     | 1888.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3558.4     | 2635.852 | 2.636  | 2134.0     | 3728.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3456.4     | 2560.321 | 2.560  | 2018.0     | 3465.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1979.3     | 1466.123 | 1.466  | 1801.0     | 1956.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3533.2     | 2617.185 | 2.617  | 2129.0     | 3601.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1913.5     | 1417.432 | 1.417  | 1833.0     | 1908.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3493.3     | 2587.605 | 2.588  | 2057.0     | 3513.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3555.3     | 2633.580 | 2.634  | 3533.2                | 3555.6                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2689.3     | 1992.086 | 1.992  | 1892.5                | 2685.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2709.0     | 2006.660 | 2.007  | 1887.1                | 2727.8                   |

Final op time: 2.634 us (3555.3 cycles), bucket=NCRISC gather

