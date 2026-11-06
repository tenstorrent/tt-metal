# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_27_36
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

Report: generated/profiler/reports/2026_11_04_05_27_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1645.1     | 1218.593 | 1.219  | 1590.0     | 1631.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.6       | 19.679   | 0.020  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1627.5     | 1205.531 | 1.206  | 1461.0     | 1605.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1759.6     | 1303.383 | 1.303  | 1593.0     | 1742.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.1      | 423.037  | 0.423  | 428.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 365.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1752.7     | 1298.272 | 1.298  | 1698.0     | 1730.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.9       | 14.025   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1600.8     | 1185.802 | 1.186  | 1437.0     | 1593.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1738.1     | 1287.481 | 1.287  | 1581.0     | 1738.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.5      | 423.358  | 0.423  | 428.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.556  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1698.9     | 1258.432 | 1.258  | 1645.1                | 1698.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.8       | 16.852   | 0.017  | 18.9                  | 22.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1614.2     | 1195.667 | 1.196  | 1600.8                | 1614.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1748.8     | 1295.432 | 1.295  | 1738.1                | 1748.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.3      | 423.198  | 0.423  | 571.1                 | 571.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.716  | 0.274  | 369.3                 | 369.5                    |

Final op time: 1.295 us (1748.8 cycles), bucket=TRISC_0 compute

