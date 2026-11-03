# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_32_52
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

Report: generated/profiler/reports/2026_11_02_09_33_13/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3881.0     | 2874.790 | 2.875  | 2637.0     | 3838.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2160.6     | 1600.469 | 1.600  | 2106.0     | 2142.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3793.7     | 2810.173 | 2.810  | 2676.0     | 3747.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3912.5     | 2898.123 | 2.898  | 2691.0     | 3840.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3852.7     | 2853.877 | 2.854  | 2623.0     | 3763.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2176.6     | 1612.296 | 1.612  | 2134.0     | 2174.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3841.4     | 2845.481 | 2.845  | 2669.0     | 3802.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3730.2     | 2763.111 | 2.763  | 2602.0     | 3727.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2242.3     | 1660.938 | 1.661  | 2129.0     | 2216.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3816.9     | 2827.358 | 2.827  | 2654.0     | 3818.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2228.2     | 1650.494 | 1.650  | 2171.0     | 2225.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3792.1     | 2808.988 | 2.809  | 2510.0     | 3780.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3862.9     | 2861.438 | 2.861  | 3816.9                | 3861.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2992.9     | 2216.988 | 2.217  | 2160.6                | 2979.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3001.2     | 2223.099 | 2.223  | 2176.6                | 3017.2                   |

Final op time: 2.861 us (3862.9 cycles), bucket=NCRISC gather

