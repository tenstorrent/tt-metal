# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_46_40
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

Report: generated/profiler/reports/2026_11_02_09_46_53/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1632.2     | 1209.062 | 1.209  | 1575.0     | 1621.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.1       | 18.568   | 0.019  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1573.6     | 1165.605 | 1.166  | 1408.0     | 1567.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1708.0     | 1265.210 | 1.265  | 1535.0     | 1702.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 563.5      | 417.407  | 0.417  | 404.0      | 567.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.4      | 278.074  | 0.278  | 367.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1706.6     | 1264.173 | 1.264  | 1652.0     | 1698.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.333   | 0.015  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1559.9     | 1155.481 | 1.155  | 1382.0     | 1545.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1701.3     | 1260.222 | 1.260  | 1536.0     | 1690.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.7      | 415.358  | 0.415  | 417.0      | 564.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.605  | 0.278  | 370.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1669.4     | 1236.617 | 1.237  | 1632.2                | 1669.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.9       | 16.951   | 0.017  | 20.7                  | 22.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1566.7     | 1160.543 | 1.161  | 1559.9                | 1566.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1704.7     | 1262.716 | 1.263  | 1701.3                | 1704.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 562.1      | 416.383  | 0.416  | 560.7                 | 562.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.1      | 277.840  | 0.278  | 374.8                 | 375.1                    |

Final op time: 1.263 us (1704.7 cycles), bucket=TRISC_0 compute

