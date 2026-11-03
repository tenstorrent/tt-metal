# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_15_05
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

Report: generated/profiler/reports/2026_11_02_09_15_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1634.5     | 1210.741 | 1.211  | 1572.0     | 1633.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.5       | 18.148   | 0.018  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1597.4     | 1183.259 | 1.183  | 1405.0     | 1605.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1730.1     | 1281.580 | 1.282  | 1524.0     | 1736.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 562.4      | 416.593  | 0.417  | 450.0      | 565.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.7      | 278.321  | 0.278  | 370.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1728.1     | 1280.074 | 1.280  | 1688.0     | 1727.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.9       | 15.481   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1588.3     | 1176.543 | 1.177  | 1329.0     | 1582.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1722.1     | 1275.630 | 1.276  | 1450.0     | 1723.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 557.0      | 412.617  | 0.413  | 404.0      | 562.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.4      | 277.333  | 0.277  | 370.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1681.3     | 1245.407 | 1.245  | 1634.5                | 1681.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.7       | 16.815   | 0.017  | 20.9                  | 22.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1592.9     | 1179.901 | 1.180  | 1588.3                | 1592.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1726.1     | 1278.605 | 1.279  | 1722.1                | 1726.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 559.7      | 414.605  | 0.415  | 557.0                 | 559.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.1      | 277.827  | 0.278  | 374.4                 | 375.1                    |

Final op time: 1.279 us (1726.1 cycles), bucket=TRISC_0 compute

