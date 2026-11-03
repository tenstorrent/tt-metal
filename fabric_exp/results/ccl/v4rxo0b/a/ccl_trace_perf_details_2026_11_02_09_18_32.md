# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_18_32
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

Report: generated/profiler/reports/2026_11_02_09_18_52/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3872.5     | 2868.543 | 2.869  | 2615.0     | 3855.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2189.7     | 1621.975 | 1.622  | 2100.0     | 2165.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3799.4     | 2814.370 | 2.814  | 2578.0     | 3823.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3893.3     | 2883.926 | 2.884  | 2635.0     | 3861.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3833.4     | 2839.580 | 2.840  | 2556.0     | 3760.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2196.0     | 1626.691 | 1.627  | 2128.0     | 2183.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3867.4     | 2864.716 | 2.865  | 2581.0     | 4187.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3749.2     | 2777.160 | 2.777  | 2593.0     | 3746.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2290.1     | 1696.395 | 1.696  | 2156.0     | 2275.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3832.7     | 2839.012 | 2.839  | 2393.0     | 3852.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2247.1     | 1664.494 | 1.664  | 2177.0     | 2243.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3806.0     | 2819.284 | 2.819  | 2546.0     | 3814.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3866.5     | 2864.049 | 2.864  | 3832.7                | 3869.9                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3004.8     | 2225.802 | 2.226  | 2189.7                | 2998.1                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3022.9     | 2239.185 | 2.239  | 2196.0                | 3044.8                   |

Final op time: 2.864 us (3866.5 cycles), bucket=NCRISC gather

