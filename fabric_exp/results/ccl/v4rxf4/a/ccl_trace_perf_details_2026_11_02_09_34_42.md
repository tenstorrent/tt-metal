# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_34_42
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

Report: generated/profiler/reports/2026_11_02_09_34_59/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1616.7     | 1197.580 | 1.198  | 1574.0     | 1606.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.1       | 20.074   | 0.020  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1554.7     | 1151.654 | 1.152  | 1161.0     | 1568.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1690.5     | 1252.198 | 1.252  | 1308.0     | 1699.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 565.3      | 418.716  | 0.419  | 405.0      | 570.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.654  | 0.278  | 370.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1696.6     | 1256.765 | 1.257  | 1672.0     | 1691.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.6       | 15.284   | 0.015  | 18.0       | 18.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1563.7     | 1158.272 | 1.158  | 1430.0     | 1514.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1699.4     | 1258.790 | 1.259  | 1552.0     | 1649.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 559.0      | 414.049  | 0.414  | 393.0      | 564.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.2      | 277.926  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1656.7     | 1227.173 | 1.227  | 1616.7                | 1656.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.9       | 17.679   | 0.018  | 20.6                  | 23.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1559.2     | 1154.963 | 1.155  | 1554.7                | 1559.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1694.9     | 1255.494 | 1.255  | 1690.5                | 1694.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 562.1      | 416.383  | 0.416  | 559.0                 | 562.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.0      | 277.790  | 0.278  | 374.8                 | 375.0                    |

Final op time: 1.255 us (1694.9 cycles), bucket=TRISC_0 compute

