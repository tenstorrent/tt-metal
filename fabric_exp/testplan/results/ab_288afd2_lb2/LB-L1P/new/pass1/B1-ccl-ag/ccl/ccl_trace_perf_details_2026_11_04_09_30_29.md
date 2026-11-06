# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_30_29
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

Report: generated/profiler/reports/2026_11_04_09_30_48/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3793.8     | 2810.247 | 2.810  | 2634.0     | 3699.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2152.3     | 1594.296 | 1.594  | 2095.0     | 2155.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3723.3     | 2757.975 | 2.758  | 2536.0     | 3707.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3823.9     | 2832.494 | 2.832  | 2608.0     | 3779.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3750.8     | 2778.395 | 2.778  | 2542.0     | 3703.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2191.3     | 1623.160 | 1.623  | 2155.0     | 2190.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3774.2     | 2795.728 | 2.796  | 2546.0     | 3838.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3667.2     | 2716.469 | 2.716  | 2599.0     | 3659.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2232.7     | 1653.852 | 1.654  | 2146.0     | 2213.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3740.6     | 2770.790 | 2.771  | 2444.0     | 3767.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 2116.2     | 1567.531 | 1.568  | 2069.0     | 2113.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3713.7     | 2750.889 | 2.751  | 2562.0     | 3703.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3783.1     | 2802.315 | 2.802  | 3740.6                | 3784.0                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2921.6     | 2164.173 | 2.164  | 2116.2                | 2909.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2965.2     | 2196.469 | 2.196  | 2191.3                | 2973.2                   |

Final op time: 2.802 us (3783.1 cycles), bucket=NCRISC gather

