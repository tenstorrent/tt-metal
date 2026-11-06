# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_33_54
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_09_34_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1733.5     | 1284.049 | 1.284  | 1693.0     | 1730.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.6       | 13.753   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1583.7     | 1173.086 | 1.173  | 1288.0     | 1596.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1722.1     | 1275.654 | 1.276  | 1425.0     | 1736.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.6      | 422.667  | 0.423  | 428.0      | 573.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.556  | 0.274  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1678.8     | 1243.556 | 1.244  | 1596.0     | 1686.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 35.4       | 26.222   | 0.026  | 19.0       | 35.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1646.3     | 1219.506 | 1.220  | 1522.0     | 1614.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1775.6     | 1315.259 | 1.315  | 1655.0     | 1743.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 576.9      | 427.333  | 0.427  | 443.0      | 580.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.358  | 0.273  | 363.0      | 369.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1706.1     | 1263.802 | 1.264  | 1678.8                | 1706.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 27.0       | 19.988   | 0.020  | 18.6                  | 27.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1615.0     | 1196.296 | 1.196  | 1583.7                | 1615.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1748.9     | 1295.457 | 1.295  | 1722.1                | 1748.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.8      | 425.000  | 0.425  | 570.6                 | 573.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.457  | 0.273  | 369.0                 | 369.2                    |

Final op time: 1.295 us (1748.9 cycles), bucket=TRISC_0 compute

