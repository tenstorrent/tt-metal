# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_38_25
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

Report: generated/profiler/reports/2026_11_04_05_38_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1655.3     | 1226.173 | 1.226  | 1591.0     | 1653.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 32.2       | 23.877   | 0.024  | 18.0       | 34.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1593.0     | 1180.025 | 1.180  | 1357.0     | 1568.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1730.6     | 1281.926 | 1.282  | 1481.0     | 1717.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.5      | 423.333  | 0.423  | 416.0      | 576.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.654  | 0.274  | 362.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1728.2     | 1280.123 | 1.280  | 1668.0     | 1694.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.7       | 14.593   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1590.2     | 1177.951 | 1.178  | 1372.0     | 1565.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1728.1     | 1280.099 | 1.280  | 1499.0     | 1700.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.2      | 420.148  | 0.420  | 436.0      | 571.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1691.8     | 1253.148 | 1.253  | 1655.3                | 1691.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 26.0       | 19.235   | 0.019  | 19.7                  | 26.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1591.6     | 1178.988 | 1.179  | 1590.2                | 1591.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1729.4     | 1281.012 | 1.281  | 1728.1                | 1729.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.4      | 421.741  | 0.422  | 567.2                 | 569.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.543  | 0.274  | 369.1                 | 369.3                    |

Final op time: 1.281 us (1729.4 cycles), bucket=TRISC_0 compute

