# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_32_18
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

Report: generated/profiler/reports/2026_11_04_09_32_36/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3394.2     | 2514.247 | 2.514  | 2105.0     | 3374.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1938.7     | 1436.049 | 1.436  | 1884.0     | 1925.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3347.0     | 2479.259 | 2.479  | 2364.0     | 3352.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3423.2     | 2535.679 | 2.536  | 2348.0     | 3362.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3325.6     | 2463.432 | 2.463  | 2041.0     | 3290.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1855.6     | 1374.494 | 1.374  | 1831.0     | 1851.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3382.5     | 2505.531 | 2.506  | 2180.0     | 3351.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3287.8     | 2435.383 | 2.435  | 2300.0     | 3268.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1869.8     | 1385.062 | 1.385  | 1833.0     | 1867.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3367.2     | 2494.247 | 2.494  | 2300.0     | 3354.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1909.0     | 1414.049 | 1.414  | 1856.0     | 1903.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3311.2     | 2452.765 | 2.453  | 2047.0     | 3288.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3391.8     | 2512.426 | 2.512  | 3367.2                | 3388.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2615.3     | 1937.228 | 1.937  | 1909.0                | 2613.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2595.9     | 1922.895 | 1.923  | 1855.6                | 2590.5                   |

Final op time: 2.512 us (3391.8 cycles), bucket=NCRISC gather

