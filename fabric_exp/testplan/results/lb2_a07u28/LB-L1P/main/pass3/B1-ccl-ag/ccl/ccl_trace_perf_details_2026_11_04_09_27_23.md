# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_23
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

Report: generated/profiler/reports/2026_11_04_09_27_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3507.5     | 2598.148 | 2.598  | 2224.0     | 3403.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1954.2     | 1447.580 | 1.448  | 1868.0     | 1963.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3466.4     | 2567.704 | 2.568  | 2247.0     | 3436.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3545.1     | 2626.000 | 2.626  | 2288.0     | 3513.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3446.8     | 2553.185 | 2.553  | 2146.0     | 3419.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1891.7     | 1401.259 | 1.401  | 1788.0     | 1891.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3495.9     | 2589.556 | 2.590  | 2257.0     | 3501.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3392.8     | 2513.185 | 2.513  | 2152.0     | 3412.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1917.2     | 1420.148 | 1.420  | 1801.0     | 1913.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3474.5     | 2573.679 | 2.574  | 2094.0     | 3462.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1928.2     | 1428.321 | 1.428  | 1854.0     | 1928.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3430.7     | 2541.284 | 2.541  | 2132.0     | 3434.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3505.7     | 2596.846 | 2.597  | 3474.5                | 3501.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2680.5     | 1985.568 | 1.986  | 1928.2                | 2673.5                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2676.5     | 1982.599 | 1.983  | 1891.7                | 2674.0                   |

Final op time: 2.597 us (3505.7 cycles), bucket=NCRISC gather

