# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_35_31
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_04_09_35_44/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1737.1     | 1286.765 | 1.287  | 1693.0     | 1733.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.5       | 13.728   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1620.4     | 1200.272 | 1.200  | 1521.0     | 1631.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1713.1     | 1268.938 | 1.269  | 1622.0     | 1724.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.8      | 422.049  | 0.422  | 457.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.6      | 273.012  | 0.273  | 360.0      | 368.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1669.2     | 1236.469 | 1.236  | 1604.0     | 1674.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 32.4       | 23.975   | 0.024  | 18.0       | 30.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1666.3     | 1234.321 | 1.234  | 1550.0     | 1637.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1760.5     | 1304.099 | 1.304  | 1633.0     | 1729.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.9      | 425.136  | 0.425  | 424.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.580  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1703.2     | 1261.617 | 1.262  | 1669.2                | 1703.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.5       | 18.852   | 0.019  | 18.5                  | 25.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1643.3     | 1217.296 | 1.217  | 1620.4                | 1643.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1736.8     | 1286.519 | 1.287  | 1713.1                | 1736.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.8      | 423.593  | 0.424  | 569.8                 | 571.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.9      | 273.296  | 0.273  | 368.6                 | 368.9                    |

Final op time: 1.287 us (1736.8 cycles), bucket=TRISC_0 compute

