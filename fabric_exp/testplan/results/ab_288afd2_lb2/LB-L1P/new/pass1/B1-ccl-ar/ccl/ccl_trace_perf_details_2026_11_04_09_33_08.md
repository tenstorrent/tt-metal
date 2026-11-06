# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_33_08
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

Report: generated/profiler/reports/2026_11_04_09_33_25/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1650.2     | 1222.370 | 1.222  | 1593.0     | 1651.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.1       | 21.580   | 0.022  | 18.0       | 25.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1594.8     | 1181.333 | 1.181  | 1517.0     | 1597.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1729.5     | 1281.136 | 1.281  | 1653.0     | 1729.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.7      | 424.247  | 0.424  | 433.0      | 576.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.309  | 0.273  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1723.6     | 1276.765 | 1.277  | 1685.0     | 1715.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.7       | 13.877   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1581.5     | 1171.481 | 1.171  | 1451.0     | 1558.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1713.2     | 1269.012 | 1.269  | 1583.0     | 1687.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.9      | 424.346  | 0.424  | 403.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.580  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1686.9     | 1249.568 | 1.250  | 1650.2                | 1686.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.9       | 17.728   | 0.018  | 18.7                  | 23.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1588.2     | 1176.407 | 1.176  | 1581.5                | 1588.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1721.3     | 1275.074 | 1.275  | 1713.2                | 1721.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.8      | 424.296  | 0.424  | 572.7                 | 572.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.1      | 273.444  | 0.273  | 369.0                 | 369.1                    |

Final op time: 1.275 us (1721.3 cycles), bucket=TRISC_0 compute

