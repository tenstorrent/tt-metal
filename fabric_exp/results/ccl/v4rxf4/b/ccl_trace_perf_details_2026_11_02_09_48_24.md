# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_48_24
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

Report: generated/profiler/reports/2026_11_02_09_48_36/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1625.4     | 1203.975 | 1.204  | 1576.0     | 1611.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.5       | 18.889   | 0.019  | 18.0       | 23.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1604.7     | 1188.691 | 1.189  | 1505.0     | 1610.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1698.7     | 1258.272 | 1.258  | 1601.0     | 1704.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 580.1      | 429.704  | 0.430  | 383.0      | 590.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.4      | 277.333  | 0.277  | 369.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1701.3     | 1260.222 | 1.260  | 1661.0     | 1692.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.9       | 15.481   | 0.015  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1592.0     | 1179.259 | 1.179  | 1427.0     | 1566.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1688.7     | 1250.864 | 1.251  | 1520.0     | 1664.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.3      | 420.198  | 0.420  | 414.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.2      | 277.901  | 0.278  | 368.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1663.3     | 1232.099 | 1.232  | 1625.4                | 1663.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.2       | 17.185   | 0.017  | 20.9                  | 23.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1598.4     | 1183.975 | 1.184  | 1592.0                | 1598.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1693.7     | 1254.568 | 1.255  | 1688.7                | 1693.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.7      | 424.951  | 0.425  | 567.3                 | 573.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.617  | 0.278  | 374.4                 | 374.8                    |

Final op time: 1.255 us (1693.7 cycles), bucket=TRISC_0 compute

