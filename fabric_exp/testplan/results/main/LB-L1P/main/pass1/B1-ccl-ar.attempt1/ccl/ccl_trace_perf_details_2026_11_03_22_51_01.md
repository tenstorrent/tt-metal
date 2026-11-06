# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_51_01
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

Report: generated/profiler/reports/2026_11_03_22_51_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1649.4     | 1221.778 | 1.222  | 1589.0     | 1646.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.5       | 19.654   | 0.020  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1673.7     | 1239.802 | 1.240  | 1483.0     | 1674.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1765.2     | 1307.556 | 1.308  | 1574.0     | 1776.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.7      | 424.247  | 0.424  | 441.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.284  | 0.273  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1772.7     | 1313.086 | 1.313  | 1692.0     | 1743.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.444   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1650.1     | 1222.321 | 1.222  | 1425.0     | 1631.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1742.5     | 1290.716 | 1.291  | 1519.0     | 1716.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.2      | 423.827  | 0.424  | 436.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.728  | 0.274  | 356.0      | 371.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1711.0     | 1267.432 | 1.267  | 1649.4                | 1711.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.0       | 17.049   | 0.017  | 19.5                  | 23.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1661.9     | 1231.062 | 1.231  | 1650.1                | 1661.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1753.8     | 1299.136 | 1.299  | 1742.5                | 1753.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.5      | 424.037  | 0.424  | 572.2                 | 572.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.506  | 0.274  | 368.9                 | 369.2                    |

Final op time: 1.299 us (1753.8 cycles), bucket=TRISC_0 compute

