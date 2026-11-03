# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_59_03
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

Report: generated/profiler/reports/2026_11_02_09_59_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 4055.0     | 3003.728 | 3.004  | 2714.0     | 4005.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2207.0     | 1634.815 | 1.635  | 2108.0     | 2178.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3979.2     | 2947.580 | 2.948  | 2705.0     | 3964.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 4071.6     | 3016.000 | 3.016  | 2905.0     | 4011.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 4005.6     | 2967.086 | 2.967  | 2685.0     | 3946.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2211.2     | 1637.926 | 1.638  | 2115.0     | 2204.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 4007.2     | 2968.272 | 2.968  | 2742.0     | 4001.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3905.9     | 2893.235 | 2.893  | 2579.0     | 3921.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2288.7     | 1695.333 | 1.695  | 2125.0     | 2253.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 4003.0     | 2965.185 | 2.965  | 2852.0     | 4005.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2265.5     | 1678.148 | 1.678  | 2155.0     | 2271.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3945.4     | 2922.494 | 2.922  | 2518.0     | 3986.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 4034.2     | 2988.296 | 2.988  | 4003.0                | 4031.1                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 3096.0     | 2293.321 | 2.293  | 2207.0                | 3085.7                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 3106.1     | 2300.833 | 2.301  | 2211.2                | 3117.0                   |

Final op time: 2.988 us (4034.2 cycles), bucket=NCRISC gather

