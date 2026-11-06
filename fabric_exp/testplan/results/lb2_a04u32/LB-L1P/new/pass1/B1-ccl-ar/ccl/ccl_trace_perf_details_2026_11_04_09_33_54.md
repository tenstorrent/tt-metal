# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_33_54
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

Report: generated/profiler/reports/2026_11_04_09_34_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1638.3     | 1213.580 | 1.214  | 1597.0     | 1627.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.4       | 19.556   | 0.020  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1621.2     | 1200.864 | 1.201  | 1195.0     | 1629.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1713.7     | 1269.407 | 1.269  | 1289.0     | 1729.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.7      | 426.444  | 0.426  | 457.0      | 577.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.704  | 0.274  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1726.2     | 1278.642 | 1.279  | 1672.0     | 1729.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.1       | 14.914   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1624.0     | 1202.938 | 1.203  | 1436.0     | 1571.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1716.9     | 1271.802 | 1.272  | 1527.0     | 1668.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.7      | 423.506  | 0.424  | 470.0      | 573.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.8      | 273.160  | 0.273  | 363.0      | 368.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1682.2     | 1246.111 | 1.246  | 1638.3                | 1682.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.3       | 17.235   | 0.017  | 20.1                  | 23.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1622.6     | 1201.901 | 1.202  | 1621.2                | 1622.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1715.3     | 1270.605 | 1.271  | 1713.7                | 1715.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.7      | 424.975  | 0.425  | 571.7                 | 573.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.1      | 273.432  | 0.273  | 368.8                 | 369.1                    |

Final op time: 1.271 us (1715.3 cycles), bucket=TRISC_0 compute

