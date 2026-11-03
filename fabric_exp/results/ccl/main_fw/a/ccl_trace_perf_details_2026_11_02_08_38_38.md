# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_08_38_38
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

Report: generated/profiler/reports/2026_11_02_08_38_53/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3562.6     | 2638.963 | 2.639  | 2134.0     | 3552.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1844.5     | 1366.296 | 1.366  | 1752.0     | 1817.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3496.2     | 2589.778 | 2.590  | 2080.0     | 3520.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3604.1     | 2669.704 | 2.670  | 2323.0     | 3495.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3520.1     | 2607.457 | 2.607  | 2072.0     | 3427.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1874.9     | 1388.840 | 1.389  | 1792.0     | 1858.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3536.0     | 2619.259 | 2.619  | 2262.0     | 3468.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3435.2     | 2544.617 | 2.545  | 2128.0     | 3540.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1986.1     | 1471.210 | 1.471  | 1845.0     | 1958.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3516.6     | 2604.864 | 2.605  | 2189.0     | 3553.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1914.5     | 1418.148 | 1.418  | 1812.0     | 1925.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3462.4     | 2564.716 | 2.565  | 2111.0     | 3371.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3554.8     | 2633.198 | 2.633  | 3516.6                | 3549.3                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2678.6     | 1984.130 | 1.984  | 1844.5                | 2674.9                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2704.9     | 2003.636 | 2.004  | 1874.9                | 2724.2                   |

Final op time: 2.633 us (3554.8 cycles), bucket=NCRISC gather

