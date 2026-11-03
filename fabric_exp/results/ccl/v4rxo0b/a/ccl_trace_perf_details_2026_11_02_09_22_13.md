# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_22_13
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_22_26/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1631.8     | 1208.716 | 1.209  | 1574.0     | 1613.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.6       | 21.210   | 0.021  | 18.0       | 24.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1604.6     | 1188.568 | 1.189  | 1406.0     | 1599.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1696.6     | 1256.765 | 1.257  | 1521.0     | 1694.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 576.7      | 427.160  | 0.427  | 416.0      | 582.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.802  | 0.278  | 370.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1706.4     | 1264.000 | 1.264  | 1665.0     | 1699.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.1       | 14.864   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1595.2     | 1181.654 | 1.182  | 1443.0     | 1574.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1691.3     | 1252.815 | 1.253  | 1549.0     | 1670.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.7      | 420.519  | 0.421  | 418.0      | 579.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.605  | 0.278  | 368.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1669.1     | 1236.358 | 1.236  | 1631.8                | 1669.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.4       | 18.037   | 0.018  | 20.1                  | 24.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1599.9     | 1185.111 | 1.185  | 1595.2                | 1599.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1694.0     | 1254.790 | 1.255  | 1691.3                | 1694.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.2      | 423.840  | 0.424  | 567.7                 | 572.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.704  | 0.278  | 374.8                 | 374.9                    |

Final op time: 1.255 us (1694.0 cycles), bucket=TRISC_0 compute

