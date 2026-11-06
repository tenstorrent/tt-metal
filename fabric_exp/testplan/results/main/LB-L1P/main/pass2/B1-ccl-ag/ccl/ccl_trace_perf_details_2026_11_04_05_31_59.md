# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_31_59
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

Report: generated/profiler/reports/2026_11_04_05_32_08/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4036.6     | 2990.099 | 2.990  | 2746.0     | 3998.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2186.3     | 1619.457 | 1.619  | 2096.0     | 2182.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3954.7     | 2929.407 | 2.929  | 3083.0     | 3944.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4066.8     | 3012.469 | 3.012  | 3081.0     | 4005.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3987.2     | 2953.506 | 2.954  | 2718.0     | 3927.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2218.2     | 1643.136 | 1.643  | 2139.0     | 2213.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 4015.9     | 2974.741 | 2.975  | 2753.0     | 3950.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3912.9     | 2898.469 | 2.898  | 3078.0     | 3907.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2299.1     | 1703.012 | 1.703  | 2121.0     | 2282.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3991.7     | 2956.840 | 2.957  | 3068.0     | 3977.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2262.1     | 1675.654 | 1.676  | 2151.0     | 2271.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3945.1     | 2922.296 | 2.922  | 2652.0     | 3966.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 4027.8     | 2983.537 | 2.984  | 3991.7                | 4026.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3087.1     | 2286.772 | 2.287  | 2186.3                | 3087.5                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3104.3     | 2299.463 | 2.299  | 2218.2                | 3122.1                   |

Final op time: 2.984 us (4027.8 cycles), bucket=NCRISC gather

