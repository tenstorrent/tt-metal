# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_55_13
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

Report: generated/profiler/reports/2026_11_02_09_55_26/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1629.3     | 1206.889 | 1.207  | 1574.0     | 1621.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.1       | 18.593   | 0.019  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1606.5     | 1190.025 | 1.190  | 1516.0     | 1607.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1699.3     | 1258.765 | 1.259  | 1613.0     | 1692.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.7      | 427.951  | 0.428  | 418.0      | 588.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.6      | 278.198  | 0.278  | 372.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1698.6     | 1258.222 | 1.258  | 1658.0     | 1693.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.0       | 14.840   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1584.3     | 1173.531 | 1.174  | 1463.0     | 1575.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1678.9     | 1243.654 | 1.244  | 1557.0     | 1667.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 565.6      | 418.988  | 0.419  | 442.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.630  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1663.9     | 1232.556 | 1.233  | 1629.3                | 1663.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.6       | 16.716   | 0.017  | 20.0                  | 22.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1595.4     | 1181.778 | 1.182  | 1584.3                | 1595.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1689.1     | 1251.210 | 1.251  | 1678.9                | 1689.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.7      | 423.469  | 0.423  | 565.6                 | 571.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.2      | 277.914  | 0.278  | 374.8                 | 375.2                    |

Final op time: 1.251 us (1689.1 cycles), bucket=TRISC_0 compute

