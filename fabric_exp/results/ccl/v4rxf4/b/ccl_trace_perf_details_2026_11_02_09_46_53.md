# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_46_53
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_47_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1620.0     | 1199.975 | 1.200  | 1573.0     | 1609.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.9       | 16.198   | 0.016  | 18.0       | 19.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1554.0     | 1151.136 | 1.151  | 1157.0     | 1561.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1683.8     | 1247.259 | 1.247  | 1277.0     | 1696.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.0      | 420.765  | 0.421  | 450.0      | 571.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.704  | 0.278  | 370.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1698.3     | 1257.975 | 1.258  | 1653.0     | 1692.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.4       | 15.852   | 0.016  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1559.4     | 1155.136 | 1.155  | 1408.0     | 1512.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1692.7     | 1253.877 | 1.254  | 1545.0     | 1649.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 564.5      | 418.148  | 0.418  | 438.0      | 565.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.728  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1659.1     | 1228.975 | 1.229  | 1620.0                | 1659.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.6       | 16.025   | 0.016  | 21.4                  | 21.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1556.7     | 1153.136 | 1.153  | 1554.0                | 1556.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1688.3     | 1250.568 | 1.251  | 1683.8                | 1688.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 566.3      | 419.457  | 0.419  | 564.5                 | 566.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.716  | 0.278  | 374.9                 | 374.9                    |

Final op time: 1.251 us (1688.3 cycles), bucket=TRISC_0 compute

