# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_08_38_06
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_08_38_21/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3559.8     | 2636.889 | 2.637  | 2094.0     | 3515.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1852.1     | 1371.926 | 1.372  | 1752.0     | 1826.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3473.8     | 2573.160 | 2.573  | 2060.0     | 3546.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3587.5     | 2657.383 | 2.657  | 2079.0     | 3846.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3502.2     | 2594.222 | 2.594  | 1987.0     | 3474.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1875.7     | 1389.383 | 1.389  | 1790.0     | 1873.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3569.5     | 2644.074 | 2.644  | 2104.0     | 4097.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3461.1     | 2563.802 | 2.564  | 2114.0     | 3456.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1983.3     | 1469.086 | 1.469  | 1819.0     | 1945.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3520.6     | 2607.827 | 2.608  | 1747.0     | 3541.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1903.3     | 1409.827 | 1.410  | 1799.0     | 1906.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3477.7     | 2576.074 | 2.576  | 1976.0     | 3494.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3559.3     | 2636.543 | 2.637  | 3520.6                | 3564.7                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2679.7     | 1984.944 | 1.985  | 1852.1                | 2682.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2702.6     | 2001.926 | 2.002  | 1875.7                | 2728.5                   |

Final op time: 2.637 us (3559.3 cycles), bucket=NCRISC gather

