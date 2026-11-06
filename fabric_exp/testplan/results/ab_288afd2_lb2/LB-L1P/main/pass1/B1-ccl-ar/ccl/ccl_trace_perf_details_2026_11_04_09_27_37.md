# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_37
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_09_27_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1640.7     | 1215.309 | 1.215  | 1594.0     | 1628.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.5       | 18.123   | 0.018  | 18.0       | 21.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1597.4     | 1183.259 | 1.183  | 1224.0     | 1602.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1730.8     | 1282.074 | 1.282  | 1358.0     | 1732.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.5      | 427.802  | 0.428  | 428.0      | 581.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.1      | 272.691  | 0.273  | 364.0      | 368.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1741.1     | 1289.704 | 1.290  | 1654.0     | 1708.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.444   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1596.2     | 1182.346 | 1.182  | 1391.0     | 1559.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1732.6     | 1283.383 | 1.283  | 1523.0     | 1694.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.8      | 421.333  | 0.421  | 428.0      | 573.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.7      | 273.136  | 0.273  | 362.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1690.9     | 1252.506 | 1.253  | 1640.7                | 1690.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.0       | 16.284   | 0.016  | 19.5                  | 22.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1596.8     | 1182.802 | 1.183  | 1596.2                | 1596.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1731.7     | 1282.728 | 1.283  | 1730.8                | 1731.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.2      | 424.568  | 0.425  | 568.8                 | 573.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.4      | 272.914  | 0.273  | 368.1                 | 368.4                    |

Final op time: 1.283 us (1731.7 cycles), bucket=TRISC_0 compute

