# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_25_59
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

Report: generated/profiler/reports/2026_11_04_09_26_20/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3963.4     | 2935.877 | 2.936  | 2687.0     | 3948.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2190.8     | 1622.840 | 1.623  | 2090.0     | 2171.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3882.6     | 2875.975 | 2.876  | 2523.0     | 3871.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3982.7     | 2950.148 | 2.950  | 2720.0     | 4016.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3900.3     | 2889.136 | 2.889  | 2535.0     | 3861.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2195.8     | 1626.494 | 1.626  | 2104.0     | 2181.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3943.5     | 2921.136 | 2.921  | 2716.0     | 4012.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3838.6     | 2843.383 | 2.843  | 2529.0     | 3825.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2310.1     | 1711.185 | 1.711  | 2107.0     | 2294.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3914.6     | 2899.728 | 2.900  | 2573.0     | 3902.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2134.7     | 1581.235 | 1.581  | 2077.0     | 2121.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3877.6     | 2872.296 | 2.872  | 2531.0     | 3890.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3951.1     | 2926.722 | 2.927  | 3914.6                | 3953.5                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3016.1     | 2234.148 | 2.234  | 2134.7                | 3014.7                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3066.5     | 2271.488 | 2.271  | 2195.8                | 3093.8                   |

Final op time: 2.927 us (3951.1 cycles), bucket=NCRISC gather

