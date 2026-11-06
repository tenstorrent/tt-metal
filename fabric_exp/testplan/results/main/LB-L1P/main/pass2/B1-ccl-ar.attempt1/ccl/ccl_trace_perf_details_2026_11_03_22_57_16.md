# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_57_16
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

Report: generated/profiler/reports/2026_11_03_22_57_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1647.8     | 1220.593 | 1.221  | 1595.0     | 1633.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.6       | 21.160   | 0.021  | 18.0       | 25.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1587.9     | 1176.222 | 1.176  | 1184.0     | 1564.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1724.1     | 1277.136 | 1.277  | 1333.0     | 1705.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.8      | 423.531  | 0.424  | 422.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.4      | 272.889  | 0.273  | 362.0      | 368.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1728.9     | 1280.691 | 1.281  | 1652.0     | 1691.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.2       | 14.988   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1620.0     | 1200.000 | 1.200  | 1406.0     | 1547.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1755.4     | 1300.296 | 1.300  | 1538.0     | 1678.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.2      | 424.568  | 0.425  | 428.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.407  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1688.4     | 1250.642 | 1.251  | 1647.8                | 1688.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.4       | 18.074   | 0.018  | 20.2                  | 24.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1604.0     | 1188.111 | 1.188  | 1587.9                | 1604.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1739.8     | 1288.716 | 1.289  | 1724.1                | 1739.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.5      | 424.049  | 0.424  | 571.8                 | 572.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.8      | 273.148  | 0.273  | 368.4                 | 368.8                    |

Final op time: 1.289 us (1739.8 cycles), bucket=TRISC_0 compute

