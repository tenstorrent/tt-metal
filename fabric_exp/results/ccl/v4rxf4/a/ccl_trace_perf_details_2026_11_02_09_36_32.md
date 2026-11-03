# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_36_32
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

Report: generated/profiler/reports/2026_11_02_09_36_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1630.7     | 1207.901 | 1.208  | 1581.0     | 1612.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.2       | 20.173   | 0.020  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1599.4     | 1184.716 | 1.185  | 1486.0     | 1594.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1691.0     | 1252.617 | 1.253  | 1572.0     | 1690.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.6      | 427.852  | 0.428  | 405.0      | 585.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.2      | 277.185  | 0.277  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1699.8     | 1259.111 | 1.259  | 1654.0     | 1692.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.0       | 15.531   | 0.016  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1587.4     | 1175.852 | 1.176  | 1478.0     | 1547.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1685.7     | 1248.642 | 1.249  | 1574.0     | 1653.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 564.4      | 418.049  | 0.418  | 421.0      | 569.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.5      | 277.407  | 0.277  | 369.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1665.2     | 1233.506 | 1.234  | 1630.7                | 1665.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.1       | 17.852   | 0.018  | 21.0                  | 24.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1593.4     | 1180.284 | 1.180  | 1587.4                | 1593.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1688.3     | 1250.630 | 1.251  | 1685.7                | 1688.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.0      | 422.951  | 0.423  | 564.4                 | 571.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.4      | 277.296  | 0.277  | 374.2                 | 374.4                    |

Final op time: 1.251 us (1688.3 cycles), bucket=TRISC_0 compute

