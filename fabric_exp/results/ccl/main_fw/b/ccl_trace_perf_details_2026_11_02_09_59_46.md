# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_59_46
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

Report: generated/profiler/reports/2026_11_02_09_59_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3579.3     | 2651.358 | 2.651  | 2081.0     | 3567.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1857.4     | 1375.827 | 1.376  | 1749.0     | 1818.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3513.8     | 2602.840 | 2.603  | 2187.0     | 3518.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3609.4     | 2673.654 | 2.674  | 2245.0     | 3639.5        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3536.2     | 2619.432 | 2.619  | 1999.0     | 3490.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1876.5     | 1390.000 | 1.390  | 1792.0     | 1882.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3579.8     | 2651.728 | 2.652  | 2081.0     | 3717.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3463.9     | 2565.852 | 2.566  | 2143.0     | 3516.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1986.4     | 1471.432 | 1.471  | 1835.0     | 1938.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3548.7     | 2628.642 | 2.629  | 2243.0     | 3574.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1895.8     | 1404.272 | 1.404  | 1799.0     | 1899.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3501.0     | 2593.309 | 2.593  | 2016.0     | 3524.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3579.3     | 2651.346 | 2.651  | 3548.7                | 3579.6                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2688.3     | 1991.346 | 1.991  | 1857.4                | 2679.8                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2719.4     | 2014.395 | 2.014  | 1876.5                | 2743.7                   |

Final op time: 2.651 us (3579.3 cycles), bucket=NCRISC gather

