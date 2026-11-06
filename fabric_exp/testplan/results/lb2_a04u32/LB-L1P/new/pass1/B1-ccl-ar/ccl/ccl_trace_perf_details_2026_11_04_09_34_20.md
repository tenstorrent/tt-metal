# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_34_20
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

Report: generated/profiler/reports/2026_11_04_09_34_32/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1657.4     | 1227.728 | 1.228  | 1598.0     | 1652.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.4       | 20.321   | 0.020  | 18.0       | 26.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1641.0     | 1215.556 | 1.216  | 1461.0     | 1638.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1731.4     | 1282.543 | 1.283  | 1547.0     | 1730.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.6      | 425.654  | 0.426  | 416.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.8      | 273.185  | 0.273  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1735.2     | 1285.358 | 1.285  | 1677.0     | 1728.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.2       | 14.988   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1631.7     | 1208.691 | 1.209  | 1518.0     | 1600.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1726.2     | 1278.667 | 1.279  | 1616.0     | 1701.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 566.1      | 419.309  | 0.419  | 442.0      | 571.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 273.975  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1696.3     | 1256.543 | 1.257  | 1657.4                | 1696.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.8       | 17.654   | 0.018  | 20.2                  | 23.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1636.4     | 1212.123 | 1.212  | 1631.7                | 1636.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1728.8     | 1280.605 | 1.281  | 1726.2                | 1728.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.4      | 422.481  | 0.422  | 566.1                 | 570.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.580  | 0.274  | 368.8                 | 369.3                    |

Final op time: 1.281 us (1728.8 cycles), bucket=TRISC_0 compute

