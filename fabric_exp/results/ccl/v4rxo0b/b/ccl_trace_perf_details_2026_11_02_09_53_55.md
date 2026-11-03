# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_53_55
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_54_07/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1627.4     | 1205.506 | 1.206  | 1574.0     | 1611.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.8       | 19.877   | 0.020  | 18.0       | 20.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1569.8     | 1162.840 | 1.163  | 1463.0     | 1565.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1698.0     | 1257.778 | 1.258  | 1595.0     | 1692.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.7      | 421.975  | 0.422  | 372.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 373.6      | 276.716  | 0.277  | 368.0      | 372.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1701.8     | 1260.568 | 1.261  | 1650.0     | 1692.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.1       | 15.605   | 0.016  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1559.0     | 1154.840 | 1.155  | 1465.0     | 1534.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1690.6     | 1252.321 | 1.252  | 1570.0     | 1670.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 559.2      | 414.247  | 0.414  | 450.0      | 562.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.654  | 0.278  | 372.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1664.6     | 1233.037 | 1.233  | 1627.4                | 1664.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.9       | 17.741   | 0.018  | 21.1                  | 23.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1564.4     | 1158.840 | 1.159  | 1559.0                | 1564.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1694.3     | 1255.049 | 1.255  | 1690.6                | 1694.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 564.5      | 418.111  | 0.418  | 559.2                 | 564.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.2      | 277.185  | 0.277  | 373.6                 | 374.2                    |

Final op time: 1.255 us (1694.3 cycles), bucket=TRISC_0 compute

