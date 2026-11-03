# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_22_00
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_22_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1621.4     | 1201.012 | 1.201  | 1576.0     | 1610.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.7       | 18.272   | 0.018  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1591.3     | 1178.716 | 1.179  | 1276.0     | 1592.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1683.5     | 1247.062 | 1.247  | 1375.0     | 1686.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 582.1      | 431.160  | 0.431  | 439.0      | 593.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.457  | 0.277  | 368.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1698.5     | 1258.173 | 1.258  | 1657.0     | 1696.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.6       | 14.494   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1591.0     | 1178.494 | 1.178  | 1431.0     | 1552.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1685.0     | 1248.123 | 1.248  | 1524.0     | 1647.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.7      | 421.259  | 0.421  | 430.0      | 577.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.0      | 277.062  | 0.277  | 367.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1659.9     | 1229.593 | 1.230  | 1621.4                | 1659.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.1       | 16.383   | 0.016  | 19.6                  | 22.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1591.1     | 1178.605 | 1.179  | 1591.0                | 1591.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1684.2     | 1247.593 | 1.248  | 1683.5                | 1684.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 575.4      | 426.210  | 0.426  | 568.7                 | 575.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.3      | 277.259  | 0.277  | 374.0                 | 374.3                    |

Final op time: 1.248 us (1684.2 cycles), bucket=TRISC_0 compute

