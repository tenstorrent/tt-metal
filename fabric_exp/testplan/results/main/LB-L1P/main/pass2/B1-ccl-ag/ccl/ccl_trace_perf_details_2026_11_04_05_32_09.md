# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_32_09
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

Report: generated/profiler/reports/2026_11_04_05_32_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3516.1     | 2604.543 | 2.605  | 2251.0     | 3550.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1857.2     | 1375.679 | 1.376  | 1782.0     | 1841.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3450.9     | 2556.247 | 2.556  | 2215.0     | 3427.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3537.7     | 2620.494 | 2.620  | 2244.0     | 3522.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3454.0     | 2558.543 | 2.559  | 2171.0     | 3404.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1897.4     | 1405.506 | 1.406  | 1802.0     | 1899.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3529.7     | 2614.617 | 2.615  | 2259.0     | 3659.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3421.6     | 2534.543 | 2.535  | 2020.0     | 3447.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1982.1     | 1468.222 | 1.468  | 1829.0     | 1945.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3504.9     | 2596.247 | 2.596  | 2153.0     | 3595.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1932.1     | 1431.210 | 1.431  | 1831.0     | 1916.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3469.4     | 2569.901 | 2.570  | 2123.0     | 3522.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3522.1     | 2608.975 | 2.609  | 3504.9                | 3522.9                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2666.2     | 1974.994 | 1.975  | 1857.2                | 2676.9                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2700.0     | 1999.969 | 2.000  | 1897.4                | 2716.5                   |

Final op time: 2.609 us (3522.1 cycles), bucket=NCRISC gather

