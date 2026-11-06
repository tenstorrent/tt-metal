# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_50_09
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

Report: generated/profiler/reports/2026_11_03_22_50_25/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1645.3     | 1218.741 | 1.219  | 1590.0     | 1638.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.5       | 20.346   | 0.020  | 18.0       | 24.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1594.3     | 1180.963 | 1.181  | 1407.0     | 1596.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1723.6     | 1276.765 | 1.277  | 1544.0     | 1718.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.7      | 426.469  | 0.426  | 434.0      | 578.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.852  | 0.274  | 366.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1721.8     | 1275.407 | 1.275  | 1654.0     | 1721.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.4       | 14.395   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1583.0     | 1172.568 | 1.173  | 1387.0     | 1572.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1716.1     | 1271.160 | 1.271  | 1523.0     | 1703.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.9      | 425.086  | 0.425  | 432.0      | 576.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.333  | 0.273  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1683.5     | 1247.074 | 1.247  | 1645.3                | 1683.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.4       | 17.370   | 0.017  | 19.4                  | 23.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1588.6     | 1176.765 | 1.177  | 1583.0                | 1588.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1719.8     | 1273.963 | 1.274  | 1716.1                | 1719.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.8      | 425.778  | 0.426  | 573.9                 | 574.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.593  | 0.274  | 369.0                 | 369.4                    |

Final op time: 1.274 us (1719.8 cycles), bucket=TRISC_0 compute

