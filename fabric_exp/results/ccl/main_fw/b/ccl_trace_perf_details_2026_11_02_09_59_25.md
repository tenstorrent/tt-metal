# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_59_25
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

Report: generated/profiler/reports/2026_11_02_09_59_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3542.7     | 2624.247 | 2.624  | 2221.0     | 3472.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1854.6     | 1373.802 | 1.374  | 1760.0     | 1820.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3484.9     | 2581.407 | 2.581  | 2162.0     | 3508.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3567.0     | 2642.222 | 2.642  | 2282.0     | 3560.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3487.3     | 2583.210 | 2.583  | 2134.0     | 3480.0        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1873.4     | 1387.728 | 1.388  | 1800.0     | 1863.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3530.4     | 2615.086 | 2.615  | 2265.0     | 3485.5        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3418.0     | 2531.877 | 2.532  | 2276.0     | 3456.5        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1959.6     | 1451.531 | 1.452  | 1820.0     | 1947.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3508.7     | 2599.062 | 2.599  | 2235.0     | 3628.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1911.9     | 1416.247 | 1.416  | 1796.0     | 1903.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3461.2     | 2563.827 | 2.564  | 2051.0     | 3450.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3537.2     | 2620.154 | 2.620  | 3508.7                | 3536.6                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2668.0     | 1976.284 | 1.976  | 1854.6                | 2665.0                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2694.8     | 1996.123 | 1.996  | 1873.4                | 2710.4                   |

Final op time: 2.620 us (3537.2 cycles), bucket=NCRISC gather

