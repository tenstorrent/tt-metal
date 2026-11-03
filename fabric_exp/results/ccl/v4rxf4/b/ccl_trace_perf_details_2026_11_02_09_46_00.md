# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_46_00
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

Report: generated/profiler/reports/2026_11_02_09_46_13/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3432.1     | 2542.296 | 2.542  | 2084.0     | 3371.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1805.9     | 1337.704 | 1.338  | 1749.0     | 1782.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3350.0     | 2481.506 | 2.482  | 2044.0     | 3333.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3462.1     | 2564.494 | 2.564  | 2140.0     | 3642.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3393.2     | 2513.481 | 2.513  | 2028.0     | 3328.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1833.4     | 1358.099 | 1.358  | 1791.0     | 1829.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3421.8     | 2534.667 | 2.535  | 2087.0     | 3799.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3307.8     | 2450.198 | 2.450  | 2025.0     | 3276.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1934.1     | 1432.642 | 1.433  | 1816.0     | 1902.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3385.2     | 2507.531 | 2.508  | 1718.0     | 3402.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1874.9     | 1388.840 | 1.389  | 1830.0     | 1869.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3341.0     | 2474.815 | 2.475  | 2015.0     | 3360.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3425.3     | 2537.247 | 2.537  | 3385.2                | 3426.9                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2595.4     | 1922.556 | 1.923  | 1805.9                | 2591.4                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2614.6     | 1936.765 | 1.937  | 1833.4                | 2637.5                   |

Final op time: 2.537 us (3425.3 cycles), bucket=NCRISC gather

