# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_14_33
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

Report: generated/profiler/reports/2026_11_02_09_14_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1639.4     | 1214.346 | 1.214  | 1575.0     | 1631.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.0       | 19.975   | 0.020  | 18.0       | 21.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1592.0     | 1179.284 | 1.179  | 1465.0     | 1598.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1724.2     | 1277.185 | 1.277  | 1593.0     | 1727.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 563.7      | 417.556  | 0.418  | 379.0      | 566.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.2      | 277.926  | 0.278  | 370.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1719.4     | 1273.630 | 1.274  | 1631.0     | 1730.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.5       | 15.926   | 0.016  | 18.0       | 19.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1581.2     | 1171.259 | 1.171  | 1297.0     | 1570.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1715.3     | 1270.568 | 1.271  | 1427.0     | 1712.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 561.4      | 415.877  | 0.416  | 385.0      | 565.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.4      | 277.333  | 0.277  | 369.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1679.4     | 1243.988 | 1.244  | 1639.4                | 1679.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.2       | 17.951   | 0.018  | 21.5                  | 24.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1586.6     | 1175.272 | 1.175  | 1581.2                | 1586.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1719.7     | 1273.877 | 1.274  | 1715.3                | 1719.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 562.6      | 416.716  | 0.417  | 561.4                 | 562.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.630  | 0.278  | 374.4                 | 374.8                    |

Final op time: 1.274 us (1719.7 cycles), bucket=TRISC_0 compute

