# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_21_16
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_21_34/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1623.1     | 1202.272 | 1.202  | 1574.0     | 1616.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.9       | 19.951   | 0.020  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1568.5     | 1161.852 | 1.162  | 1463.0     | 1559.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1701.2     | 1260.173 | 1.260  | 1594.0     | 1689.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.9      | 420.642  | 0.421  | 443.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.3      | 277.284  | 0.277  | 370.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1696.9     | 1256.963 | 1.257  | 1652.0     | 1687.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.1       | 14.914   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1555.3     | 1152.049 | 1.152  | 1427.0     | 1528.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1691.7     | 1253.111 | 1.253  | 1561.0     | 1673.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 563.4      | 417.358  | 0.417  | 424.0      | 565.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.7      | 277.580  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1660.0     | 1229.617 | 1.230  | 1623.1                | 1660.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.432   | 0.017  | 20.1                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1561.9     | 1156.951 | 1.157  | 1555.3                | 1561.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1696.5     | 1256.642 | 1.257  | 1691.7                | 1696.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 565.6      | 419.000  | 0.419  | 563.4                 | 565.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.5      | 277.432  | 0.277  | 374.3                 | 374.5                    |

Final op time: 1.257 us (1696.5 cycles), bucket=TRISC_0 compute

