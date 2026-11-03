# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_16_02
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_16_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1637.7     | 1213.136 | 1.213  | 1571.0     | 1632.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.5       | 20.346   | 0.020  | 18.0       | 25.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1639.2     | 1214.247 | 1.214  | 1484.0     | 1632.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1736.5     | 1286.296 | 1.286  | 1586.0     | 1732.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.3      | 424.691  | 0.425  | 374.0      | 578.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.4      | 278.074  | 0.278  | 372.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1735.0     | 1285.210 | 1.285  | 1660.0     | 1726.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.0       | 15.531   | 0.016  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1625.6     | 1204.148 | 1.204  | 1392.0     | 1629.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1723.0     | 1276.272 | 1.276  | 1499.0     | 1732.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 566.6      | 419.728  | 0.420  | 451.0      | 569.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.802  | 0.278  | 369.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1686.4     | 1249.173 | 1.249  | 1637.7                | 1686.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.2       | 17.938   | 0.018  | 21.0                  | 24.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1632.4     | 1209.198 | 1.209  | 1625.6                | 1632.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1729.7     | 1281.284 | 1.281  | 1723.0                | 1729.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.0      | 422.210  | 0.422  | 566.6                 | 570.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.2      | 277.938  | 0.278  | 375.0                 | 375.2                    |

Final op time: 1.281 us (1729.7 cycles), bucket=TRISC_0 compute

