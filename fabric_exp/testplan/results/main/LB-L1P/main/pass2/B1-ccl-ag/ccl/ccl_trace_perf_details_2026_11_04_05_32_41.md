# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_32_41
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_04_05_32_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3546.4     | 2626.938 | 2.627  | 2167.0     | 3520.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1902.2     | 1409.012 | 1.409  | 1791.0     | 1844.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3479.0     | 2577.012 | 2.577  | 2082.0     | 3505.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3571.6     | 2645.654 | 2.646  | 2222.0     | 3663.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3498.4     | 2591.432 | 2.591  | 2142.0     | 3468.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1874.3     | 1388.395 | 1.388  | 1816.0     | 1871.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3556.0     | 2634.099 | 2.634  | 2105.0     | 3955.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3453.2     | 2557.951 | 2.558  | 2128.0     | 3456.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 2008.0     | 1487.383 | 1.487  | 1869.0     | 1993.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3519.3     | 2606.889 | 2.607  | 2217.0     | 3526.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1937.5     | 1435.160 | 1.435  | 1854.0     | 1930.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3492.2     | 2586.840 | 2.587  | 2079.0     | 3517.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3548.3     | 2628.395 | 2.628  | 3519.3                | 3551.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2697.8     | 1998.389 | 1.998  | 1902.2                | 2695.3                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2713.4     | 2009.907 | 2.010  | 1874.3                | 2743.5                   |

Final op time: 2.628 us (3548.3 cycles), bucket=NCRISC gather

